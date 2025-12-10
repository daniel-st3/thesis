# Local Testing Guide

Follow these steps to rerun the model comparison and feature-importance scripts on a machine with internet access.

## 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
```

## 2) Install dependencies
You can install everything listed in `environment.yml` via pip:
```bash
pip install pandas==2.2.2 numpy scikit-learn==1.5.0 imbalanced-learn matplotlib==3.9.0 seaborn==0.13.2 xgboost==2.0.3 yfinance==0.2.40 pyyaml gnews==0.2.1 praw vaderSentiment
```

If you prefer conda/mamba:
```bash
conda env create -f environment.yml
conda activate thesis_env
```

## 3) Run the scripts
Ensure the data files are present (the repo includes `data/3_final_modeling_dataset.csv`). Then run:
```bash
python scripts/4_run_final_model_comparison.py
python scripts/plot_feature_importance.py
```

Each script writes outputs to `data/model_comparison_results.(csv|pkl)` and `plots/final_thesis_plots/9_feature_importance.png` respectively.

## 4) Troubleshooting
- If you see `ModuleNotFoundError: No module named 'pandas'`, rerun step 2 to install dependencies.
- Proxy-restricted environments may block `pip`/`conda`. In that case, use a machine with normal internet access or download wheels in advance and install with `pip install --no-index --find-links <wheel_dir> ...`.
