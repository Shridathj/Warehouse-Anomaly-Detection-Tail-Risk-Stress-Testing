"""
Pytest configuration and shared fixtures for the Warehouse Anomaly Detection project.
"""

import sys
from pathlib import Path
import warnings

import pandas as pd
import pytest

# Suppress non-critical warnings during testing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def scenario_configs():
    """Load scenario configurations once per test session."""
    from src.config import SCENARIO_CONFIGS

    return SCENARIO_CONFIGS


@pytest.fixture
def minimal_dataframe() -> pd.DataFrame:
    """Create a small synthetic DataFrame for testing."""
    return pd.DataFrame(
        {
            "InvoiceNo": [str(i) for i in range(50)],
            "Quantity": [i % 8 for i in range(50)],
            "UnitPrice": [12.5 + (i * 0.3) for i in range(50)],
            "InvoiceDate": pd.date_range("2020-01-01", periods=50, freq="h"),
            "CustomerID": [f"C{i % 6}" for i in range(50)],
        }
    )