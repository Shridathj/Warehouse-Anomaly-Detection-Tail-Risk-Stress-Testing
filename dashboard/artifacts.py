"""Serialize / load dashboard stage outputs for instant viewing without re-running."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import plotly.io as pio

from dashboard.pipeline import StageOutput

CACHE_ROOT = Path(__file__).resolve().parent.parent / "results" / "dashboard_cache"


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
    return SerializedStage(
        title=stage.title,
        logs=stage.logs,
        mpl_pngs=[_fig_to_png(fig) for fig in stage.mpl_figs],
        plotly_jsons=[pio.to_json(fig) for fig in stage.plotly_figs],
    )


def deserialize_stage(data: SerializedStage) -> StageOutput:
    return StageOutput(
        title=data.title,
        logs=data.logs,
        mpl_figs=[],
        plotly_figs=[pio.from_json(j) for j in data.plotly_jsons],
        mpl_pngs=data.mpl_pngs,
    )


def _cache_path(scenario_id: int) -> Path:
    return CACHE_ROOT / f"scenario_{scenario_id}" / "stages.pkl"


def artifacts_exist(scenario_id: int) -> bool:
    return _cache_path(scenario_id).exists()


def save_artifacts(scenario_id: int, stages: list[StageOutput]) -> Path:
    path = _cache_path(scenario_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [serialize_stage(s) for s in stages]
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_artifacts(scenario_id: int) -> list[StageOutput] | None:
    path = _cache_path(scenario_id)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        payload: list[SerializedStage] = pickle.load(fh)
    return [deserialize_stage(item) for item in payload]
