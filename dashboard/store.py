"""Backend data access for bundles, documents, plots, and the retail file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dashboard.documents import DOCUMENT_SPECS, DocumentSpec, spec_by_id
from dashboard.paths import PATHS, ProjectPaths, format_bytes, relative_to_root
from dashboard.pipeline import StageOutput


_LOADER_ALIASES = {
    "1": "gross",
    "s1": "gross",
    "gross": "gross",
    "s1_gross": "gross",
    "scenario_1": "gross",
    "2": "netted",
    "s2": "netted",
    "netted": "netted",
    "s2_netted": "netted",
    "scenario_2": "netted",
}


def normalize_loader_key(scenario_key: str | int) -> str:
    key = str(scenario_key).strip().lower()
    if key not in _LOADER_ALIASES:
        raise ValueError(
            f"Unknown scenario key {scenario_key!r}. Use 'gross'/'netted' or 1/2."
        )
    return _LOADER_ALIASES[key]


def scenario_loader_key(scenario_id: int) -> str:
    return "gross" if int(scenario_id) == 1 else "netted"


@dataclass(frozen=True)
class ArtifactMeta:
    scenario_id: int
    path: str
    display_path: str
    exists: bool
    size_bytes: int
    size_label: str
    modified_iso: str | None
    source: str


@dataclass(frozen=True)
class ResolvedDocument:
    id: str
    title: str
    kicker: str
    description: str
    filename: str
    path: str
    display_path: str
    exists: bool
    size_bytes: int
    size_label: str
    searched: tuple[str, ...]


@dataclass
class StageLoadResult:
    ok: bool
    stages: list[StageOutput] = field(default_factory=list)
    meta: ArtifactMeta | None = None
    error: str | None = None
    fallback_used: bool = False
    source_label: str = ""


class DashboardStore:
    def __init__(self, paths: ProjectPaths | None = None) -> None:
        self.paths = paths or PATHS

    def artifact_meta(self, scenario_id: int) -> ArtifactMeta:
        path = self.paths.scenario_bundle(scenario_id)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        modified = None
        if exists:
            modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                timespec="seconds"
            )
        return ArtifactMeta(
            scenario_id=int(scenario_id),
            path=str(path),
            display_path=relative_to_root(path, self.paths.root),
            exists=exists,
            size_bytes=size,
            size_label=format_bytes(size) if exists else "missing",
            modified_iso=modified,
            source="bundle" if exists else "none",
        )

    def load_stages(self, scenario_id: int) -> StageLoadResult:
        from dashboard.artifacts import load_artifacts

        meta = self.artifact_meta(scenario_id)
        try:
            stages = load_artifacts(scenario_id)
        except Exception as exc:
            fallback = self._fallback_stage(scenario_id)
            if fallback:
                return StageLoadResult(
                    ok=True,
                    stages=[fallback],
                    meta=meta,
                    error=str(exc),
                    fallback_used=True,
                    source_label="disk plots (bundle unreadable)",
                )
            return StageLoadResult(
                ok=False,
                meta=meta,
                error=f"Could not read bundled cache: {exc}",
                source_label="error",
            )

        if stages:
            return StageLoadResult(
                ok=True,
                stages=stages,
                meta=meta,
                source_label=f"bundled cache · {meta.size_label}",
            )

        fallback = self._fallback_stage(scenario_id)
        if fallback:
            return StageLoadResult(
                ok=True,
                stages=[fallback],
                meta=meta,
                fallback_used=True,
                source_label="disk plots (no bundle)",
            )

        return StageLoadResult(
            ok=False,
            meta=meta,
            error=(
                f"No bundled cache at `{meta.path}`. "
                "Run the live pipeline or `python scripts/build_dashboard_artifacts.py`."
            ),
            source_label="missing",
        )

    def save_stages(self, scenario_id: int, stages: list[StageOutput]) -> ArtifactMeta:
        from dashboard.artifacts import save_artifacts

        save_artifacts(scenario_id, stages)
        return self.artifact_meta(scenario_id)

    def fallback_plot_images(self, scenario_id: int) -> list[tuple[str, bytes]]:
        out: list[tuple[str, bytes]] = []
        seen: set[str] = set()
        for folder in self.paths.scenario_plot_dirs(scenario_id):
            if not folder.is_dir():
                continue
            for png in sorted(folder.glob("*.png")):
                if png.name in seen:
                    continue
                seen.add(png.name)
                out.append((png.name, png.read_bytes()))
        return out

    def _fallback_stage(self, scenario_id: int) -> StageOutput | None:
        images = self.fallback_plot_images(scenario_id)
        if not images:
            return None
        return StageOutput(
            title="Saved pipeline charts (fallback)",
            logs=(
                "Bundled pickle was unavailable. Showing PNG charts from "
                f"{', '.join(str(p) for p in self.paths.scenario_plot_dirs(scenario_id))}."
            ),
            mpl_pngs=[blob for _name, blob in images],
        )

    def list_documents(self) -> list[ResolvedDocument]:
        return [self.resolve_document(spec.id) for spec in DOCUMENT_SPECS]

    def resolve_document(self, doc_id: str) -> ResolvedDocument:
        spec = spec_by_id(doc_id)
        if spec is None:
            return ResolvedDocument(
                id=doc_id,
                title=doc_id,
                kicker="Unknown",
                description="This document is not in the dashboard catalog.",
                filename="",
                path="",
                display_path="",
                exists=False,
                size_bytes=0,
                size_label="missing",
                searched=(),
            )
        return self._resolve_spec(spec)

    def _resolve_spec(self, spec: DocumentSpec) -> ResolvedDocument:
        searched: list[str] = []
        chosen: Path | None = None
        for rel in spec.relatives:
            path = self.paths.root / rel
            searched.append(str(path))
            if path.exists() and chosen is None:
                chosen = path
        exists = chosen is not None
        size = chosen.stat().st_size if chosen else 0
        return ResolvedDocument(
            id=spec.id,
            title=spec.title,
            kicker=spec.kicker,
            description=spec.description,
            filename=spec.filename,
            path=str(chosen) if chosen else (searched[0] if searched else ""),
            display_path=(
                relative_to_root(chosen, self.paths.root)
                if chosen
                else (searched[0] if searched else "")
            ),
            exists=exists,
            size_bytes=size,
            size_label=format_bytes(size) if exists else "missing",
            searched=tuple(searched),
        )

    def read_document_bytes(self, doc_id: str) -> bytes | None:
        doc = self.resolve_document(doc_id)
        if not doc.exists:
            return None
        return Path(doc.path).read_bytes()

    def dataset_status(self) -> dict:
        candidates = [
            self.paths.dataset / "Online Retail.xlsx",
            self.paths.dataset / "raw" / "Online Retail.xlsx",
        ]
        for path in candidates:
            if path.exists():
                size = path.stat().st_size
                return {
                    "exists": True,
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": size,
                    "size_label": format_bytes(size),
                }
        return {
            "exists": False,
            "path": None,
            "name": "Online Retail.xlsx",
            "size_bytes": 0,
            "size_label": "missing",
            "searched": [str(p) for p in candidates],
        }

    def readme_text(self) -> str:
        path = self.paths.readme
        if not path.exists():
            return "_README.md was not found in the project root._"
        return path.read_text(encoding="utf-8")

    def health_snapshot(self) -> dict:
        return {
            "dataset": self.dataset_status(),
            "artifacts": {
                "1": asdict(self.artifact_meta(1)),
                "2": asdict(self.artifact_meta(2)),
            },
            "documents": [asdict(doc) for doc in self.list_documents()],
        }


STORE = DashboardStore()
