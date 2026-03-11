import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from matminer.datasets.convenience_loaders import load_expt_gap
from matminer.featurizers.composition import ElementProperty

if __name__ == "__main__":
    np.random.seed(42)

    # featurize compositions using Magpie element-property descriptors
    df = load_expt_gap()
    df["composition"] = df["formula"].apply(Composition)
    featurizer = ElementProperty.from_preset("magpie")
    featurizer.set_n_jobs(1)
    X = featurizer.featurize_dataframe(df, "composition", ignore_errors=True)
    feature_cols = [c for c in X.columns if c not in ("formula", "gap expt", "composition")]
    X = X[feature_cols + ["gap expt"]].dropna(axis=0).reset_index(drop=True)
    y = np.asarray(X["gap expt"], dtype=float)
    X = np.asarray(X[feature_cols], dtype=float)
    print("y: min={:.2f} max={:.2f} mean={:.2f} eV".format(float(y.min()), float(y.max()), float(y.mean())))
    print(len(X), "samples", X.shape[1], "features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # drop near-zero-variance features before fitting
    vt = VarianceThreshold(threshold=1e-6)
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)
    kept = vt.get_support()
    feature_cols = [f for f, k in zip(feature_cols, kept) if k]
    print("Features:", X_train.shape[1], "(dropped", np.sum(~kept), ")")
    results = []

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    gs = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-3, 3, 13)}, cv=5, scoring="neg_mean_absolute_error")
    gs.fit(X_train, y_train)
    pred = gs.predict(X_test)
    y_pred_ridge = pred
    results.append({"model": "Ridge", "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)})
    print("Ridge", results[-1]["MAE"], results[-1]["R2"])

    pipe = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso())])
    gs = GridSearchCV(pipe, {"lasso__alpha": np.logspace(-4, 0, 17)}, cv=5, scoring="neg_mean_absolute_error")
    gs.fit(X_train, y_train)
    pred = gs.predict(X_test)
    results.append({"model": "Lasso", "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)})
    lasso_coef = gs.best_estimator_.named_steps["lasso"].coef_
    lasso_importance = pd.DataFrame({"feature": feature_cols, "coef": lasso_coef}).sort_values("coef", key=abs, ascending=False)
    print("Lasso", results[-1]["MAE"], results[-1]["R2"])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(max_iter=500, random_state=42, early_stopping=True)),
    ])
    gs = GridSearchCV(pipe, {"mlp__hidden_layer_sizes": [(64, 32), (128, 64), (256, 128)], "mlp__learning_rate_init": [0.01, 0.001], "mlp__alpha": [1e-3, 1e-2]}, cv=5, scoring="neg_mean_absolute_error")
    gs.fit(X_train, y_train)
    cv_df = pd.DataFrame(gs.cv_results_)
    param_cols = [c for c in cv_df.columns if c.startswith("param_")]
    print("\nMLP grid (higher score = better):")
    print(cv_df[param_cols + ["mean_test_score"]].sort_values("mean_test_score", ascending=False).to_string(index=False))
    pred = gs.predict(X_test)
    results.append({"model": "MLP", "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)})
    mlp_est = gs.best_estimator_
    y_pred_mlp = pred
    print("MLP", results[-1]["MAE"], results[-1]["R2"])

    # split by band gap == 0 (metals) vs > 0 (non-metals)
    metals_mask = (y_test == 0)
    nonmetals_mask = (y_test > 0)
    print("\nMAE metal vs non-metal:")
    for name, pred in [("Ridge", y_pred_ridge), ("MLP", y_pred_mlp)]:
        mae_m = np.mean(np.abs(pred[metals_mask] - y_test[metals_mask]))
        mae_nm = np.mean(np.abs(pred[nonmetals_mask] - y_test[nonmetals_mask]))
        n_m, n_nm = metals_mask.sum(), nonmetals_mask.sum()
        print("  {} | metals (n={}): {:.3f} eV | non-metals (n={}): {:.3f} eV".format(name, n_m, mae_m, n_nm, mae_nm))

    # re-train best MLP config across multiple seeds to estimate variance
    mlp_step = mlp_est.named_steps["mlp"]
    hls = mlp_step.hidden_layer_sizes
    alpha = mlp_step.alpha
    seeds = [42, 123, 7, 99, 256]
    mlp_maes = []
    for seed in seeds:
        pipe_seed = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=hls, alpha=alpha, max_iter=500, random_state=seed, early_stopping=True)),
        ])
        pipe_seed.fit(X_train, y_train)
        mlp_maes.append(mean_absolute_error(y_test, pipe_seed.predict(X_test)))
    print("\nMLP MAE (5 seeds): {:.3f} ± {:.3f} eV".format(np.mean(mlp_maes), np.std(mlp_maes)))

    # fit full PCA to find how many components explain 90% of variance
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca_full = PCA().fit(X_train_s)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = np.searchsorted(cumvar, 0.90) + 1
    print("90% variance at", n90, "components")

    for n in [10, 20, 40, n90]:
        n = min(n, X_train_s.shape[1] - 1, X_train_s.shape[0] - 1)
        if n < 2:
            continue
        pca = PCA(n_components=n)
        Xtr = pca.fit_transform(X_train_s)
        Xte = pca.transform(X_test_s)
        pipe = Pipeline([("lr", Ridge(alpha=0.1))])
        pipe.fit(Xtr, y_train)
        pred = pipe.predict(Xte)
        results.append({"model": f"Ridge PCA{n}", "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)})

    n_pca = min(25, n90, X_train_s.shape[1] - 1, X_train_s.shape[0] - 1)
    if n_pca < 2:
        n_pca = 2
    pca = PCA(n_components=n_pca)
    Xtr = pca.fit_transform(X_train_s)
    Xte = pca.transform(X_test_s)
    scaler_pca = StandardScaler()
    Xtr = scaler_pca.fit_transform(Xtr)
    Xte = scaler_pca.transform(Xte)
    for name, model in [("Ridge", Ridge()), ("Lasso", Lasso())]:
        pipe = Pipeline([("m", model)])
        gs = GridSearchCV(pipe, {"m__alpha": np.logspace(-3, 1, 9)}, cv=5, scoring="neg_mean_absolute_error")
        gs.fit(Xtr, y_train)
        pred = gs.predict(Xte)
        results.append({"model": name + " PCA", "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)})
    pipe = Pipeline([("mlp", MLPRegressor(max_iter=500, random_state=42, early_stopping=True))])
    gs = GridSearchCV(pipe, {"mlp__hidden_layer_sizes": [(64, 32)], "mlp__alpha": [1e-2]}, cv=5, scoring="neg_mean_absolute_error")
    gs.fit(Xtr, y_train)
    pred = gs.predict(Xte)
    results.append({"model": "MLP PCA", "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)})

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("\nLasso top 15")
    print(lasso_importance.head(15).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/results.csv", index=False)
    lasso_importance.to_csv("results/lasso_importance.csv", index=False)
    pd.DataFrame({"component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1), "variance_ratio": pca_full.explained_variance_ratio_}).to_csv("results/pca_variance.csv", index=False)
    pd.DataFrame({"y_true": y_test, "y_pred": mlp_est.predict(X_test)}).to_csv("results/predictions.csv", index=False)
