"""Dashboard path, document, and metric access."""

from dashboard.documents import DOCUMENT_SPECS
from dashboard.metrics import parse_headline_metrics
from dashboard.paths import project_root as resolve_root
from dashboard.store import STORE, normalize_loader_key


def test_project_root_matches_pytest_root(project_root):
    assert resolve_root() == project_root


def test_normalize_loader_key():
    assert normalize_loader_key("1") == "gross"
    assert normalize_loader_key("gross") == "gross"
    assert normalize_loader_key("s2_netted") == "netted"


def test_documents_resolve_on_disk():
    docs = STORE.list_documents()
    ids = {doc.id for doc in docs}
    assert {spec.id for spec in DOCUMENT_SPECS} <= ids
    missing = [doc.id for doc in docs if not doc.exists]
    assert missing == [], f"Missing documents: {missing}"


def test_monograph_is_catalogued():
    doc = STORE.resolve_document("evt_monograph")
    assert doc.exists
    assert doc.filename.endswith(".pdf")
    assert "EVT" in doc.title or "Extreme" in doc.description


def test_artifact_bundles_exist():
    assert STORE.artifact_meta(1).exists
    assert STORE.artifact_meta(2).exists


def test_parse_headline_metrics(project_root):
    text = (project_root / "results" / "expected_result.txt").read_text(encoding="utf-8")
    metrics = parse_headline_metrics(text)
    assert metrics.dragons == "127"
    assert metrics.es95 is not None
    assert metrics.unfulfilled == "66"
