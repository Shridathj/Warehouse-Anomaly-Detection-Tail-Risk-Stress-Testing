"""Catalog of static documents served by the dashboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSpec:
    id: str
    title: str
    kicker: str
    description: str
    filename: str
    relatives: tuple[str, ...]


DOCUMENT_SPECS: tuple[DocumentSpec, ...] = (
    DocumentSpec(
        id="anomaly_summary",
        title="Updated Anomaly Summary",
        kicker="Project report",
        description="Project report: findings, method, and scope notes.",
        filename="updated_anomaly_summary.pdf",
        relatives=(
            "static/updated_anomaly_summary.pdf",
            "project_report/updated_anomaly_summary.pdf",
        ),
    ),
    DocumentSpec(
        id="evt_monograph",
        title="A Comprehensive Approach to Tail-Risk Estimation via EVT",
        kicker="Research monograph",
        description="Research monograph on tail-risk estimation via extreme value theory.",
        filename="A_Comprehensive_Approach_to_Tail_Risk_Estimation_via_EVT.pdf",
        relatives=(
            "static/A_Comprehensive_Approach_to_Tail_Risk_Estimation_via_EVT.pdf",
            "project_report/A_Comprehensive_Approach_to_Tail_Risk_Estimation_via_EVT.pdf",
        ),
    ),
)


def spec_by_id(doc_id: str) -> DocumentSpec | None:
    for spec in DOCUMENT_SPECS:
        if spec.id == doc_id:
            return spec
    return None
