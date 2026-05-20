# Deploy the dashboard to the cloud (Streamlit Community Cloud)

Running `python run run_src.py` on your laptop executes the full pipeline **on your laptop**. To avoid that, deploy the app to **Streamlit Community Cloud**. The browser UI stays light; heavy work runs on Streamlit’s servers.

## 1. Push the repo to GitHub

Ensure your project is on GitHub.

## 2. Deploy on Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select this repository, branch `main` (or your default branch).
4. Set **Main file path** to: `streamlit_dashboard.py`
5. Click **Deploy**.

The dataset (`Online Retail.xlsx`) is downloaded automatically from UCI on first use if it is not in the repo.

## 3. Optional secrets

In the app → **Settings** → **Secrets**, paste:

```toml
[dashboard]
cloud_url = "https://YOUR-APP-NAME.streamlit.app"
```

Use the same URL Streamlit gives you. When you still run the app locally, the UI will show a link to the cloud app instead of running heavy jobs on your laptop.

## 4. How to use after deploy

| What you want | What to do |
|---------------|------------|
| **No laptop compute** | Open the `*.streamlit.app` URL in your browser (do not run `streamlit run` locally). |
| **Instant results** | Build bundled cache once (see below), commit `results/dashboard_cache/**/stages.pkl`, redeploy. |
| **Re-run full pipeline in cloud** | In the app sidebar → **Run live pipeline (cloud)**. |

## 5. Build bundled results (optional, recommended)

On a machine that can finish the pipeline (or once on Streamlit Cloud after a successful live run), from repo root:

```bash
pip install -r requirements.txt
python scripts/build_dashboard_artifacts.py
```

Commit:

```
results/dashboard_cache/scenario_1/stages.pkl
results/dashboard_cache/scenario_2/stages.pkl
```

Redeploy. Users then choose **View bundled results (instant, no compute)**.

## 6. Local vs cloud behaviour

| Environment | Where pipeline runs |
|-------------|---------------------|
| `streamlit run` on your PC | Your laptop (can be slow / high RAM) |
| `https://….streamlit.app` | Streamlit Cloud servers |
| Bundled results mode | Nowhere (loads pre-saved outputs) |

## Limits

Streamlit Cloud free apps have memory and CPU limits. If a live run fails with out-of-memory errors, use **bundled results** or upgrade the hosting plan.

## Troubleshooting

- **App crashes on Scenario 1**: Use bundled results, or run `build_dashboard_artifacts.py` on a stronger machine and commit the `.pkl` files.
- **Dataset missing**: The loader downloads from UCI; ensure outbound network is allowed on Streamlit Cloud (default: yes).
- **Still heavy on laptop**: You are using the local URL or clicked **Run live pipeline (this laptop)** — switch to the cloud URL.
