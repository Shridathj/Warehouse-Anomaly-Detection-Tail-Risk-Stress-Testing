"""
Streamlit dashboard entry point.

Local UI only (pipeline runs on this machine unless using bundled results):
    streamlit run streamlit_dashboard.py

For cloud compute, deploy to Streamlit Community Cloud — see DEPLOYMENT.md.
"""

from dashboard.app import main

if __name__ == "__main__":
    main()
