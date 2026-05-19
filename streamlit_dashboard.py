"""
Streamlit dashboard entry point.

Local UI only (pipeline runs on this machine unless using bundled results):
    streamlit run streamlit_dashboard.py

For cloud compute, deploy to Streamlit Community Cloud — see DEPLOYMENT.md.
"""

import sys
from pathlib import Path

# Ensure repo root is in sys.path so both 'dashboard' and 'src' packages are importable
# (required for Streamlit Cloud and any clean Python environment)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import main

if __name__ == "__main__":
    main()