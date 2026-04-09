## Reproducing the Results

### Requirements
- Python 3.12+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Shridathj/Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing.git
cd Warehouse-Anomaly-Detection-Tail-Risk-Stress-Testing
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

The project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

1. Visit the link above and download `Online Retail.xlsx`.
2. Place the file in the `data/` folder:

```
data/
└── Online Retail.xlsx
```

### 5. Run the analysis

Use the top-level runner to execute both scenarios end-to-end:

```bash
python run_src.py
```

Or run each scenario individually:

```bash
python anomaly-detection-scenario1.py   # Gross (maximum exposure)
python anomaly-detection-scenario2.py   # Realistic netted exposure
```

### 6. Outputs

After running, results are written to:

```
results/
├── scenario1/
│   └── plots/
└── scenario2/
    └── plots/
reports/
expected_result.txt
```

Compare your `expected_result.txt` against the reference in the repo to verify your run is correctly calibrated.

### 7. Explore the notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
```

### Verified environment

| Package | Version |
|---|---|
| Python | 3.12 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| statsmodels | 0.14.6 |
| matplotlib | 3.10.8 |
| plotly | 6.6.0 |
| numba | 0.65.0 |
| scikit-learn | 1.8.0 |
| seaborn | 0.13.2 |
| openpyxl | 3.1.5 |

> All results were produced on this exact stack. Minor numerical differences may occur across operating systems due to floating-point handling in Monte Carlo sampling.
