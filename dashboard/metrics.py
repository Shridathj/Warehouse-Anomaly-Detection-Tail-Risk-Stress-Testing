"""Pull headline figures out of captured pipeline logs."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

from dashboard.pipeline import StageOutput

_FIRST = (
    ("transactions", (r"Raw rows\s*:\s*([\d,]+)",)),
    ("dragons", (r"Dragon events\s*:\s*(\d+)",)),
    (
        "unfulfilled",
        (
            r"Unfulfilled Dragons \(≥\d+h SLA breach\)\s+(\d+)",
            r"Dragons at risk of unfulfillment\s*:\s*(\d+)",
            r"Unfulfilled Dragons[^\n]*:\s*(\d+)",
        ),
    ),
    ("es95", (r"Expected Shortfall 95%[^\n]*£\s*([\d,]+(?:\.\d+)?)",)),
    ("var95", (r"VaR 95% \(annual\)\s*£\s*([\d,]+(?:\.\d+)?)",)),
    ("hill_xi", (r"Hill xi\s*:\s*([\d.]+)", r"Hill ξ\s*:\s*([\d.]+)")),
    ("gross_revenue", (r"Gross Revenue \(firm demand\)\s*£\s*([\d,]+(?:\.\d+)?)",)),
    ("backtest", (r"Backtest:\s*(\d+/\d+ violations[^\n]*)",)),
    ("preventable", (r"preventable dragon loss ≈ £\s*([\d,]+(?:\.\d+)?)",)),
)


@dataclass(frozen=True)
class HeadlineMetrics:
    transactions: str | None = None
    dragons: str | None = None
    unfulfilled: str | None = None
    es95: str | None = None
    var95: str | None = None
    hill_xi: str | None = None
    gross_revenue: str | None = None
    backtest: str | None = None
    preventable: str | None = None

    def any(self) -> bool:
        return any(getattr(self, f.name) for f in fields(self))


def parse_headline_metrics(text: str) -> HeadlineMetrics:
    found: dict[str, str] = {}
    for key, patterns in _FIRST:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                found[key] = match.group(1).strip()
                break
    return HeadlineMetrics(**found)


def metrics_from_stages(stages: list[StageOutput]) -> HeadlineMetrics:
    blob = "\n".join(stage.logs or "" for stage in stages)
    return parse_headline_metrics(blob)
