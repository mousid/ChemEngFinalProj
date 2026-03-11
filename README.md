# Comparing Regression Models for Materials Property Prediction

Band gap prediction from chemical composition using linear regression, Ridge, Lasso, PCA, and MLP (multilayer perceptron). Data from the **Materials Project**-related experimental band gap dataset via [matminer](https://github.com/hackingmaterials/matminer).

## Setup

```bash
cd /Users/sidg/materials-prediction
python -m venv venv   # if not already created
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_experiments.py
```

- **Data**: Matminer `expt_gap` (experimental band gap, eV). By default the script uses 2000 samples for speed; set `MAX_SAMPLES = None` in `run_experiments.py` to use the full dataset (~6354).
- **Features**: Compositional descriptors from the Magpie preset (atomic numbers, electronegativity, radius, valence, etc.) — 132 features after featurization.
- **Split**: 80% train / 20% test; 5-fold CV on the training set for hyperparameters (Ridge/Lasso alpha, MLP architecture, PCA n_components).
- **Metrics**: Test set **MAE** (eV) and **R²**.

## Output

- **Models**: Linear regression (baseline), Ridge, Lasso, MLP (2 hidden layers), each with and without PCA.
- **PCA**: Explained-variance analysis and performance for several n_components (e.g. 10, 20, 40, 90% variance).
- **Lasso**: Top features by absolute coefficient (feature importance).
- **Research questions**: (1) Can linear models predict band gap reasonably well? (2) Does PCA help or hurt? (3) Do neural networks outperform linear models?

## Files

- `run_experiments.py` — Load data, featurize, train/evaluate all models, print summary and Lasso feature importance.
- `requirements.txt` — Python dependencies.
