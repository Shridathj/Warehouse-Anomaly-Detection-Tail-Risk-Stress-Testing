"""Serialize / load dashboard stage outputs for instant viewing without re-running."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import plotly.io as pio

from dashboard.paths import PATHS
from dashboard.pipeline import StageOutput

CACHE_ROOT = PATHS.dashboard_cache


@dataclass
class SerializedStage:
    title: str
    logs: str
    mpl_pngs: list[bytes]
    plotly_jsons: list[str]


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def serialize_stage(stage: StageOutput) -> SerializedStage:
    pngs = list(stage.mpl_pngs or [])
    pngs.extend(_fig_to_png(fig) for fig in stage.mpl_figs)
    return SerializedStage(
        title=stage.title,
        logs=stage.logs,
        mpl_pngs=pngs,
        plotly_jsons=[pio.to_json(fig) for fig in stage.plotly_figs],
    )


def deserialize_stage(data: SerializedStage) -> StageOutput:
    plotly_figs = []
    for raw in getattr(data, "plotly_jsons", None) or []:
        try:
            plotly_figs.append(pio.from_json(raw))
        except Exception:
            continue
    return StageOutput(
        title=getattr(data, "title", "Stage"),
        logs=getattr(data, "logs", "") or "",
        mpl_figs=[],
        plotly_figs=plotly_figs,
        mpl_pngs=list(getattr(data, "mpl_pngs", None) or []),
    )


def _cache_path(scenario_id: int) -> Path:
    return PATHS.scenario_bundle(int(scenario_id))


def artifacts_exist(scenario_id: int) -> bool:
    return _cache_path(scenario_id).exists()


def save_artifacts(scenario_id: int, stages: list[StageOutput]) -> Path:
    path = _cache_path(scenario_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [serialize_stage(s) for s in stages]
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _coerce_payload(payload):
    if isinstance(payload, dict) and "stages" in payload:
        return payload["stages"]
    return payload


def load_artifacts(scenario_id: int) -> list[StageOutput] | None:
    path = _cache_path(scenario_id)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    items = _coerce_payload(payload)
    return [deserialize_stage(item) for item in items]
