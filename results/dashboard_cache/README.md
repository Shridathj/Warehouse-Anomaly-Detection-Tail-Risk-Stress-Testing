# Dashboard bundled results

Pre-computed pipeline outputs for instant viewing (no heavy compute in the browser session).

## Build once (powerful machine or Streamlit Cloud shell)

```bash
python scripts/build_dashboard_artifacts.py
```

This writes:

- `scenario_1/stages.pkl`
- `scenario_2/stages.pkl`

Commit those `.pkl` files if you want every user (including Streamlit Cloud) to load results instantly.

If they are missing, the cloud app can still **Run live pipeline on server** once.
