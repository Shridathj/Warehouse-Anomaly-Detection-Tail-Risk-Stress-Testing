# Pipeline plots (generated)

This folder is **wiped and recreated** when you run:

```bash
python run_src.py
```

Outputs are organised as:

- `scenario1/` — Scenario 1 (gross / max risk)
- `scenario2/` — Scenario 2 (netted)

Streamlit live runs write into the matching `scenario{N}/` subfolder (that subfolder is cleared for the scenario being run).

Do not commit large PNG batches unless needed; add `plots/**/*.png` to `.gitignore` if desired.
