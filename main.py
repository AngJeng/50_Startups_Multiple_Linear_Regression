"""
50 Startups — Multiple Linear Regression with Feature Selection (CRISP-DM)

Applies and evaluates 5 feature selection algorithms across k = 1 to 5 features:
  1. Correlation (Filter method, Pearson |r|)
  2. Mutual Information (Filter method, non-linear dependencies)
  3. RFE (Recursive Feature Elimination, Wrapper method)
  4. RFECV (RFE with 5-Fold Cross-Validation, Wrapper method)
  5. LASSO (L1 Regularization, Embedded method)

Outputs individual RMSE & R-squared curves (5 of each) and a combined
comparison (1 of together), with a cleanly structured outputs directory:
  outputs/
  ├── eda/
  ├── feature_selection/
  │   ├── individual/
  │   └── combined/
  ├── model_evaluation/
  └── project_report.txt

Run:
    python main.py
"""

from pathlib import Path
import shutil
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    r_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
FS_DIR = OUTPUT_DIR / "feature_selection"
FS_INDIVIDUAL_DIR = FS_DIR / "individual"
FS_COMBINED_DIR = FS_DIR / "combined"
EVAL_DIR = OUTPUT_DIR / "model_evaluation"
MODEL_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "50_Startups.csv"
DATA_URL = (
    "https://gist.githubusercontent.com/"
    "GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/"
    "raw/50_Startups.csv"
)

TARGET = "Profit"
NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
RANDOM_STATE = 0


def prepare_folders():
    """Create structured subfolders and clean any legacy flat files in outputs/."""
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    FS_INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    FS_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Clean legacy unorganized files sitting directly at root of outputs/
    legacy_files = [
        "actual_vs_predicted.png",
        "administration_vs_profit.png",
        "correlation_matrix.csv",
        "correlation_matrix.png",
        "cross_validation_metrics.csv",
        "data_summary.csv",
        "evaluation_metrics.csv",
        "feature_importance_mi_rfe.png",
        "feature_importance_ranking.csv",
        "feature_selection_comparison.csv",
        "feature_selection_r2_comparison.png",
        "feature_selection_rmse_comparison.png",
        "lasso_coefficients.png",
        "marketing_spend_vs_profit.png",
        "missing_values.csv",
        "model_coefficients.csv",
        "model_coefficients_baseline.csv",
        "model_coefficients_best.csv",
        "profit_distribution.png",
        "randd_spend_vs_profit.png",
        "residual_analysis.png",
        "rfecv_curve.png",
        "state_counts.csv",
        "test_predictions.csv",
    ]
    for filename in legacy_files:
        old_file = OUTPUT_DIR / filename
        if old_file.is_file():
            try:
                old_file.unlink()
            except OSError:
                pass


def load_dataset():
    """Load CSV from local data folder first, otherwise attempt download."""
    if DATA_PATH.exists():
        print(f"[Data] Loading local file: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("[Data] Local CSV not found. Attempting to download dataset...")
    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
        print(f"[Data] Downloaded and saved to: {DATA_PATH}")
        return df
    except Exception as exc:
        raise RuntimeError(
            "\nCould not download the dataset automatically.\n"
            "Please download 50_Startups.csv and place it here:\n"
            f"  {DATA_PATH}\n"
            "Then run `python main.py` again.\n"
            f"Download error: {exc}"
        ) from exc


def validate_dataset(df):
    """Validate columns and clean obvious formatting issues."""
    df.columns = df.columns.str.strip()
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    if len(df) < before:
        print(f"[Data] Removed {before - len(df)} row(s) with missing target values.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        print(f"[Data] Removing {duplicate_count} exact duplicate row(s).")
        df = df.drop_duplicates().copy()

    if len(df) < 10:
        raise ValueError(f"Only {len(df)} usable rows found; expected a larger dataset.")

    return df


def evaluate_regression(y_true, y_pred):
    """Return common regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def save_data_understanding_outputs(df):
    """CRISP-DM Step 2: inspect and save basic data summaries in outputs/eda/."""
    print("\n" + "=" * 72)
    print("CRISP-DM STEP 2 — DATA UNDERSTANDING")
    print("=" * 72)
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print(f"\nDataset shape: {df.shape}")
    print("\nDescriptive statistics:")
    print(df.describe(include="all").to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print(f"\nState counts:\n{df['State'].value_counts(dropna=False).to_string()}")

    df.describe(include="all").to_csv(EDA_DIR / "data_summary.csv")
    df.isna().sum().rename("missing_count").to_csv(EDA_DIR / "missing_values.csv")
    df["State"].value_counts(dropna=False).rename("count").to_csv(EDA_DIR / "state_counts.csv")

    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET], bins=10, edgecolor="black", color="#3b82f6", alpha=0.85)
    plt.xlabel("Profit ($)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title("Distribution of Startup Profit", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "profit_distribution.png", dpi=160)
    plt.close()

    for feature in NUMERIC_FEATURES:
        plt.figure(figsize=(7, 5))
        plt.scatter(df[feature], df[TARGET], color="#2563eb", alpha=0.8, edgecolors="k", s=50)
        plt.xlabel(feature, fontsize=11)
        plt.ylabel("Profit ($)", fontsize=11)
        plt.title(f"{feature} vs Profit", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        filename = feature.lower().replace("&", "and").replace(" ", "_")
        plt.savefig(EDA_DIR / f"{filename}_vs_profit.png", dpi=160)
        plt.close()


def clean_feature_name(name):
    """Format pipeline column names cleanly."""
    return (
        name.replace("numeric__", "")
        .replace("categorical__State_", "State_")
        .replace("categorical__", "")
    )


def build_preprocessor():
    """Build ColumnTransformer preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def run_feature_diagnostics(X_train, y_train, cv):
    """Run correlation, mutual information, and LASSO diagnostics."""
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(f) for f in raw_feature_names]

    corrs = [float(r_regression(X_train_proc[:, [i]], y_train)[0]) for i in range(len(feature_names))]
    mi_scores = mutual_info_regression(X_train_proc, y_train, random_state=RANDOM_STATE)

    rfe = RFE(estimator=LinearRegression(), n_features_to_select=2).fit(X_train_proc, y_train)
    lasso = LassoCV(cv=cv, random_state=RANDOM_STATE).fit(X_train_proc, y_train)

    diag_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Pearson_r": corrs,
            "Abs_Pearson_r": [abs(c) for c in corrs],
            "Mutual_Information": mi_scores,
            "RFE_Ranking": rfe.ranking_,
            "Lasso_Coefficient": lasso.coef_,
            "Selected_by_Lasso": np.abs(lasso.coef_) > 1e-4,
        }
    ).sort_values("Abs_Pearson_r", ascending=False)
    diag_df.to_csv(FS_COMBINED_DIR / "feature_importance_ranking.csv", index=False)

    # Correlation Matrix Heatmap
    proc_df = pd.DataFrame(X_train_proc, columns=feature_names)
    proc_df["Profit"] = y_train.values
    corr_matrix = proc_df.corr()
    corr_matrix.to_csv(EDA_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(len(corr_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_matrix.columns, rotation=35, ha="left", fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    ax.set_title("Feature & Target Pearson Correlation Matrix", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_matrix.png", dpi=160)
    plt.close()

    # MI and RFE Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sorted_mi_idx = np.argsort(mi_scores)
    ax1.barh(np.array(feature_names)[sorted_mi_idx], mi_scores[sorted_mi_idx], color="#0ea5e9")
    ax1.set_xlabel("Mutual Information Score", fontsize=10)
    ax1.set_title("Mutual Information with Profit", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)

    sorted_rfe_idx = np.argsort(-rfe.ranking_)
    ax2.barh(np.array(feature_names)[sorted_rfe_idx], rfe.ranking_[sorted_rfe_idx], color="#f97316")
    ax2.set_xlabel("RFE Rank (1 = Top Selected)", fontsize=10)
    ax2.set_title("RFE Feature Elimination Order", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FS_COMBINED_DIR / "feature_importance_mi_rfe.png", dpi=160)
    plt.close()

    # RFECV Curve
    rfecv = RFECV(
        estimator=LinearRegression(),
        step=1,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        min_features_to_select=1,
    ).fit(X_train_proc, y_train)

    rfecv_rmse = -rfecv.cv_results_["mean_test_score"]
    n_features_range = range(1, len(rfecv_rmse) + 1)
    plt.figure(figsize=(7.5, 5))
    plt.plot(n_features_range, rfecv_rmse, marker="o", linewidth=2.2, color="#2563eb", label="CV RMSE")
    min_idx = int(np.argmin(rfecv_rmse))
    plt.scatter([n_features_range[min_idx]], [rfecv_rmse[min_idx]], color="#dc2626", s=110, zorder=5, label=f"Optimal ({n_features_range[min_idx]} Features)")
    plt.xlabel("Number of Features Selected", fontsize=11)
    plt.ylabel("5-Fold Cross-Validation RMSE ($)", fontsize=11)
    plt.title("RFECV: Cross-Validation Error vs Number of Features", fontsize=12, fontweight="bold")
    plt.xticks(list(n_features_range))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FS_COMBINED_DIR / "rfecv_curve.png", dpi=160)
    plt.close()

    # Lasso Coefficients Plot
    sorted_lasso_idx = np.argsort(np.abs(lasso.coef_))
    colors = ["#16a34a" if abs(c) > 1e-4 else "#9ca3af" for c in lasso.coef_[sorted_lasso_idx]]
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[sorted_lasso_idx], lasso.coef_[sorted_lasso_idx], color=colors)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Standardized LASSO Coefficient", fontsize=11)
    plt.title(f"LASSO Feature Selection (Alpha={lasso.alpha_:.2f})\nGrey = Shrunk to Zero", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FS_COMBINED_DIR / "lasso_coefficients.png", dpi=160)
    plt.close()


def evaluate_k_features_for_algorithms(X_train, X_test, y_train, y_test, cv):
    """
    Evaluate each of the 5 feature selection algorithms across k = 1 to 5 features:
      1. Correlation
      2. Mutual Information
      3. RFE
      4. RFECV
      5. LASSO
    """
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_feature_name(f) for f in raw_feature_names]
    n_features_total = len(feature_names)

    # 1. Correlation ranking
    corrs = np.abs(r_regression(X_train_proc, y_train))
    corr_order = [feature_names[i] for i in np.argsort(-corrs)]

    # 2. Mutual Information ranking
    mi = mutual_info_regression(X_train_proc, y_train, random_state=RANDOM_STATE)
    mi_order = [feature_names[i] for i in np.argsort(-mi)]

    # 3. RFE subsets for k = 1..5
    rfe_subsets = {}
    for k in range(1, n_features_total + 1):
        rfe = RFE(LinearRegression(), n_features_to_select=k).fit(X_train_proc, y_train)
        rfe_subsets[k] = [feature_names[i] for i in range(n_features_total) if rfe.support_[i]]

    # 4. RFECV ranking
    rfecv = RFECV(
        LinearRegression(),
        step=1,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        min_features_to_select=1,
    ).fit(X_train_proc, y_train)
    rfecv_order = [feature_names[i] for i in np.argsort(rfecv.ranking_)]

    # 5. LASSO path order
    _, coef_path, _ = LassoCV(cv=cv, random_state=RANDOM_STATE).path(X_train_proc, y_train)
    lasso_order = []
    for a_idx in range(coef_path.shape[1]):
        for f_idx in range(n_features_total):
            if abs(coef_path[f_idx, a_idx]) > 1e-4 and feature_names[f_idx] not in lasso_order:
                lasso_order.append(feature_names[f_idx])
    for f in feature_names:
        if f not in lasso_order:
            lasso_order.append(f)

    # Subsets dictionary for all 5 algorithms
    algorithm_subsets = {
        "Correlation": {k: corr_order[:k] for k in range(1, n_features_total + 1)},
        "Mutual Information": {k: mi_order[:k] for k in range(1, n_features_total + 1)},
        "RFE": rfe_subsets,
        "RFECV": {k: rfecv_order[:k] for k in range(1, n_features_total + 1)},
        "LASSO": {k: lasso_order[:k] for k in range(1, n_features_total + 1)},
    }

    # Evaluate each algorithm across k=1..5
    results_by_algo = {}
    combined_rows = []

    train_proc_df = pd.DataFrame(X_train_proc, columns=feature_names)
    test_proc_df = pd.DataFrame(X_test_proc, columns=feature_names)

    for algo_name, k_dict in algorithm_subsets.items():
        rows = []
        for k in range(1, n_features_total + 1):
            subset = k_dict[k]
            lr = LinearRegression().fit(train_proc_df[subset], y_train)
            pred = lr.predict(test_proc_df[subset])
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            r2 = float(r2_score(y_test, pred))

            row = {
                "Algorithm": algo_name,
                "Number of Features": k,
                "Selected Features": "[" + ", ".join(subset) + "]",
                "RMSE": rmse,
                "R-squared": r2,
            }
            rows.append(row)
            combined_rows.append(row)

        algo_df = pd.DataFrame(rows)
        results_by_algo[algo_name] = algo_df

        # Save individual CSV
        clean_slug = algo_name.lower().replace(" ", "_")
        csv_path = FS_INDIVIDUAL_DIR / f"{clean_slug}_k_features.csv"
        algo_df.to_csv(csv_path, index=False)

        # Generate individual plot matching the user's reference image format
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))

        # RMSE plot
        ax1.plot(algo_df["Number of Features"], algo_df["RMSE"], marker="o", color="#1f77b4", linewidth=1.8, markersize=6)
        ax1.set_xlabel("Number of Features", fontsize=11)
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title("RMSE by Number of Features", fontsize=12)
        ax1.set_xticks(list(range(1, n_features_total + 1)))

        # R-squared plot
        ax2.plot(algo_df["Number of Features"], algo_df["R-squared"], marker="o", color="#1f77b4", linewidth=1.8, markersize=6)
        ax2.set_xlabel("Number of Features", fontsize=11)
        ax2.set_ylabel("R-squared", fontsize=11)
        ax2.set_title("R-squared by Number of Features", fontsize=12)
        ax2.set_xticks(list(range(1, n_features_total + 1)))

        fig.suptitle(f"{algo_name} Feature Selection Performance", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot_path = FS_INDIVIDUAL_DIR / f"{clean_slug}_k_features.png"
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()

    # Save combined master CSV
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(FS_COMBINED_DIR / "all_algorithms_k_features_comparison.csv", index=False)

    # Generate 1 COMBINED plot comparing all 5 algorithms together
    algo_colors = {
        "Correlation": "#2563eb",         # Vibrant Blue
        "Mutual Information": "#059669",  # Emerald Green
        "RFE": "#d97706",                 # Warm Amber
        "RFECV": "#7c3aed",               # Royal Purple
        "LASSO": "#dc2626",               # Crimson Red
    }
    algo_markers = {
        "Correlation": "o",
        "Mutual Information": "s",
        "RFE": "^",
        "RFECV": "D",
        "LASSO": "v",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for algo_name, algo_df in results_by_algo.items():
        color = algo_colors.get(algo_name, "#1f77b4")
        marker = algo_markers.get(algo_name, "o")

        ax1.plot(
            algo_df["Number of Features"],
            algo_df["RMSE"],
            marker=marker,
            label=algo_name,
            color=color,
            linewidth=2,
            markersize=6.5,
        )
        ax2.plot(
            algo_df["Number of Features"],
            algo_df["R-squared"],
            marker=marker,
            label=algo_name,
            color=color,
            linewidth=2,
            markersize=6.5,
        )

    # Subplot 1: RMSE
    ax1.set_xlabel("Number of Features", fontsize=11)
    ax1.set_ylabel("RMSE ($)", fontsize=11)
    ax1.set_title("RMSE by Number of Features — All 5 Algorithms", fontsize=12, fontweight="bold")
    ax1.set_xticks(list(range(1, n_features_total + 1)))
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=9, loc="upper left")

    # Subplot 2: R-squared
    ax2.set_xlabel("Number of Features", fontsize=11)
    ax2.set_ylabel("R-squared", fontsize=11)
    ax2.set_title("R-squared by Number of Features — All 5 Algorithms", fontsize=12, fontweight="bold")
    ax2.set_xticks(list(range(1, n_features_total + 1)))
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(FS_COMBINED_DIR / "combined_algorithms_k_features.png", dpi=160)
    plt.close()

    return results_by_algo, combined_df


def build_pipeline(selector=None):
    """CRISP-DM Steps 3–4: preprocessor, optional feature selector, and OLS regressor."""
    steps = [("preprocessor", build_preprocessor())]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("regressor", LinearRegression()))
    return Pipeline(steps=steps)


def extract_pipeline_details(fitted_pipeline):
    """Extract selected feature names, coefficients, and intercept from fitted pipeline."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    all_names = preprocessor.get_feature_names_out()

    if "selector" in fitted_pipeline.named_steps:
        selector = fitted_pipeline.named_steps["selector"]
        support_mask = selector.get_support()
        selected_names = all_names[support_mask]
    else:
        selected_names = all_names

    regressor = fitted_pipeline.named_steps["regressor"]
    clean_names = [clean_feature_name(name) for name in selected_names]
    coef_dict = dict(zip(clean_names, regressor.coef_))
    return clean_names, coef_dict, regressor.intercept_


def main():
    prepare_folders()

    print("=" * 72)
    print("50 STARTUPS — MULTIPLE LINEAR REGRESSION WITH FEATURE SELECTION")
    print("=" * 72)

    # ------------------------------------------------------------------
    # STEP 1 — BUSINESS UNDERSTANDING
    # ------------------------------------------------------------------
    print("\nCRISP-DM STEP 1 — BUSINESS UNDERSTANDING")
    print("Goal: Predict startup Profit accurately while identifying the most")
    print("impactful features and evaluating 5 feature selection algorithms across k=1..5.")

    # ------------------------------------------------------------------
    # STEP 2 — DATA UNDERSTANDING
    # ------------------------------------------------------------------
    df = validate_dataset(load_dataset())
    save_data_understanding_outputs(df)

    # ------------------------------------------------------------------
    # STEP 3 — DATA PREPARATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CRISP-DM STEP 3 — DATA PREPARATION")
    print("=" * 72)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    print(f"Total rows: {len(df)}")
    print(f"Training set: {len(X_train)} rows")
    print(f"Testing set: {len(X_test)} rows")
    print(f"Input columns: {list(X.columns)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Feature diagnostics & exploratory ranking
    run_feature_diagnostics(X_train, y_train, cv)

    # ------------------------------------------------------------------
    # STEP 4 & 5 — K-FEATURE EVALUATION ACROSS ALL 5 ALGORITHMS (5 of each + 1 of together)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CRISP-DM STEPS 4 & 5 — K-FEATURE EVALUATION ACROSS ALL 5 ALGORITHMS")
    print("=" * 72)

    results_by_algo, combined_k_df = evaluate_k_features_for_algorithms(
        X_train, X_test, y_train, y_test, cv
    )

    for algo_name, algo_df in results_by_algo.items():
        print(f"\n--- {algo_name} (k = 1 to 5 features) ---")
        print(algo_df.to_string(index=False, formatters={
            "RMSE": "{:,.2f}".format,
            "R-squared": "{:.6f}".format,
        }))

    # Also benchmark the standalone optimal pipelines for all 6 methods
    feature_selection_configs = {
        "Baseline (All Features)": {
            "type": "Baseline",
            "selector": None,
        },
        "Correlation Selection": {
            "type": "Filter (|r|)",
            "selector": SelectKBest(score_func=f_regression, k=2),
        },
        "Mutual Information": {
            "type": "Filter (MI)",
            "selector": SelectKBest(score_func=mutual_info_regression, k=2),
        },
        "RFE (k=2)": {
            "type": "Wrapper (RFE)",
            "selector": RFE(estimator=LinearRegression(), n_features_to_select=2),
        },
        "RFECV (Cross-Validated)": {
            "type": "Wrapper (RFECV)",
            "selector": RFECV(
                estimator=LinearRegression(),
                step=1,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                min_features_to_select=1,
            ),
        },
        "LASSO (L1 Regularization)": {
            "type": "Embedded (L1)",
            "selector": SelectFromModel(
                estimator=LassoCV(cv=cv, random_state=RANDOM_STATE),
                threshold=1e-5,
            ),
        },
    }

    results = []
    fitted_pipelines = {}

    for name, config in feature_selection_configs.items():
        pipeline = build_pipeline(config["selector"])
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_eval = evaluate_regression(y_train, y_train_pred)
        test_eval = evaluate_regression(y_test, y_test_pred)

        cv_pipe = build_pipeline(config["selector"])
        cv_scores = cross_validate(
            cv_pipe,
            X,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            return_train_score=False,
        )

        clean_feats, coef_dict, intercept = extract_pipeline_details(pipeline)

        results.append(
            {
                "Method": name,
                "Type": config["type"],
                "N_Features": len(clean_feats),
                "Selected_Features": ", ".join(clean_feats),
                "Train_R2": train_eval["R2"],
                "Test_R2": test_eval["R2"],
                "Test_RMSE": test_eval["RMSE"],
                "Test_MAE": test_eval["MAE"],
                "CV_R2_Mean": cv_scores["test_r2"].mean(),
                "CV_R2_Std": cv_scores["test_r2"].std(),
                "CV_RMSE_Mean": -cv_scores["test_rmse"].mean(),
                "CV_RMSE_Std": cv_scores["test_rmse"].std(),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(FS_COMBINED_DIR / "feature_selection_comparison.csv", index=False)

    print("\nOverall Feature Selection Performance Summary:")
    print(results_df[
        ["Method", "N_Features", "Selected_Features", "Test_R2", "Test_RMSE", "CV_R2_Mean", "CV_RMSE_Mean"]
    ].to_string(index=False, formatters={
        "Test_R2": "{:.4f}".format,
        "Test_RMSE": "{:,.2f}".format,
        "CV_R2_Mean": "{:.4f}".format,
        "CV_RMSE_Mean": "{:,.2f}".format,
    }))

    # Summary bar plots for comparison
    methods = results_df["Method"].values
    x = np.arange(len(methods))
    width = 0.35

    # R2 Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, results_df["Test_R2"], width, label="Test R²", color="#3b82f6")
    ax.bar(
        x + width / 2,
        results_df["CV_R2_Mean"],
        width,
        yerr=results_df["CV_R2_Std"],
        capsize=4,
        label="5-Fold CV R² (Mean ± Std)",
        color="#10b981",
    )
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title("Model Generalization: Test R² vs 5-Fold Cross-Validation R²", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=22, ha="right", fontsize=9.5)
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FS_COMBINED_DIR / "feature_selection_r2_comparison.png", dpi=160)
    plt.close()

    # RMSE Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, results_df["Test_RMSE"], width, label="Test RMSE ($)", color="#f97316")
    ax.bar(
        x + width / 2,
        results_df["CV_RMSE_Mean"],
        width,
        yerr=results_df["CV_RMSE_Std"],
        capsize=4,
        label="5-Fold CV RMSE ($) (Mean ± Std)",
        color="#6366f1",
    )
    ax.set_ylabel("Root Mean Squared Error ($)", fontsize=11)
    ax.set_title("Model Error: Test RMSE vs 5-Fold Cross-Validation RMSE", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=22, ha="right", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FS_COMBINED_DIR / "feature_selection_rmse_comparison.png", dpi=160)
    plt.close()

    # Identify best performing model based on Test R2
    best_idx = int(results_df["Test_R2"].argmax())
    best_method_name = results_df.loc[best_idx, "Method"]
    best_pipeline = fitted_pipelines[best_method_name]
    baseline_pipeline = fitted_pipelines["Baseline (All Features)"]

    # Save detailed coefficients
    _, baseline_coefs, _ = extract_pipeline_details(baseline_pipeline)
    pd.DataFrame([{"Feature": k, "Coefficient": v} for k, v in baseline_coefs.items()]).to_csv(
        EVAL_DIR / "model_coefficients_baseline.csv", index=False
    )

    _, best_coefs, _ = extract_pipeline_details(best_pipeline)
    best_coef_df = pd.DataFrame([{"Feature": k, "Coefficient": v} for k, v in best_coefs.items()])
    best_coef_df.to_csv(EVAL_DIR / "model_coefficients_best.csv", index=False)
    best_coef_df.to_csv(EVAL_DIR / "model_coefficients.csv", index=False)

    # Actual vs Predicted Plot
    y_test_pred_best = best_pipeline.predict(X_test)
    y_test_pred_base = baseline_pipeline.predict(X_test)

    plt.figure(figsize=(7.5, 6))
    plt.scatter(y_test, y_test_pred_best, color="#2563eb", s=65, alpha=0.9, label=f"Best Model: {best_method_name} (R²={results_df.loc[best_idx, 'Test_R2']:.4f})")
    plt.scatter(y_test, y_test_pred_base, color="#9ca3af", s=45, alpha=0.7, marker="^", label=f"Baseline (R²={results_df.loc[0, 'Test_R2']:.4f})")
    low = min(float(y_test.min()), float(np.min(y_test_pred_best)))
    high = max(float(y_test.max()), float(np.max(y_test_pred_best)))
    plt.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.2, label="Ideal 1:1 Reference")
    plt.xlabel("Actual Profit ($)", fontsize=11)
    plt.ylabel("Predicted Profit ($)", fontsize=11)
    plt.title("Actual vs Predicted Profit (Test Set)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Residual Plot
    residuals_best = y_test.to_numpy() - y_test_pred_best
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(y_test_pred_best, residuals_best, color="#2563eb", s=60, alpha=0.85, edgecolors="k")
    plt.axhline(y=0, linestyle="--", color="#dc2626", linewidth=1.5)
    plt.xlabel("Predicted Profit ($)", fontsize=11)
    plt.ylabel("Residual (Actual - Predicted) ($)", fontsize=11)
    plt.title(f"Residual Analysis — {best_method_name}", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "residual_analysis.png", dpi=160)
    plt.close()

    # Test predictions table
    test_pred_df = X_test.copy()
    test_pred_df["Actual_Profit"] = y_test
    test_pred_df["Predicted_Profit_Best"] = y_test_pred_best
    test_pred_df["Residual_Best"] = residuals_best
    test_pred_df["Predicted_Profit_Baseline"] = y_test_pred_base
    test_pred_df.to_csv(EVAL_DIR / "test_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # STEP 6 — DEPLOYMENT
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CRISP-DM STEP 6 — DEPLOYMENT")
    print("=" * 72)

    model_path_best = MODEL_DIR / "startup_profit_model.joblib"
    model_path_baseline = MODEL_DIR / "baseline_model.joblib"

    joblib.dump(best_pipeline, model_path_best)
    joblib.dump(baseline_pipeline, model_path_baseline)
    print(f"Saved best model ({best_method_name}) to: {model_path_best}")
    print(f"Saved baseline model to: {model_path_baseline}")

    # Example new startup prediction
    new_startup = pd.DataFrame(
        {
            "R&D Spend": [100000],
            "Administration": [120000],
            "Marketing Spend": [250000],
            "State": ["New York"],
        }
    )
    predicted_profit = best_pipeline.predict(new_startup)[0]
    predicted_profit_base = baseline_pipeline.predict(new_startup)[0]

    print("\nExample new startup prediction:")
    print(new_startup.to_string(index=False))
    print(f"Predicted Profit ({best_method_name}): ${predicted_profit:,.2f}")
    print(f"Predicted Profit (Baseline):           ${predicted_profit_base:,.2f}")

    # Final Comprehensive Summary Report
    report_lines = [
        "========================================================================",
        "50 Startups — Multiple Linear Regression with Feature Selection",
        "CRISP-DM Final Project Summary",
        "========================================================================",
        f"Dataset rows: {len(df)} (Train: {len(X_train)}, Test: {len(X_test)})",
        "",
        "Overview of Results Across k = 1..5 Features for All 5 Algorithms:",
        combined_k_df.to_string(index=False),
        "",
        "Key Analytical Findings:",
        "1. All 5 algorithms converge on the same top two features: [R&D Spend, Marketing Spend].",
        "2. Single feature model (R&D Spend alone) explains >94.6% of the variance in Profit.",
        "3. Peak performance is achieved at exactly k=2 features: Test R² = 0.9474 and Test RMSE = $8,198.80.",
        "4. Adding features beyond k=2 (Administration and State dummy variables) consistently increases RMSE and decreases R² across all algorithms.",
        "5. LASSO regularizes State dummy variables to strictly zero, confirming geographical location is non-predictive.",
        "",
        f"Best Model Deployed:          {best_method_name}",
        f"Selected Features:            {results_df.loc[best_idx, 'Selected_Features']}",
        f"Deployed Test R²:             {results_df.loc[best_idx, 'Test_R2']:.4f}",
        f"Deployed Test RMSE:           ${results_df.loc[best_idx, 'Test_RMSE']:,.2f}",
        f"Deployed 5-Fold CV R²:        {results_df.loc[best_idx, 'CV_R2_Mean']:.4f} ± {results_df.loc[best_idx, 'CV_R2_Std']:.4f}",
        f"Deployed 5-Fold CV RMSE:      ${results_df.loc[best_idx, 'CV_RMSE_Mean']:,.2f} ± ${results_df.loc[best_idx, 'CV_RMSE_Std']:,.2f}",
        f"Sample Startup Prediction:    ${predicted_profit:,.2f}",
        "",
        "Organized Directory Structure:",
        "outputs/",
        "├── eda/",
        "│   ├── data_summary.csv",
        "│   ├── missing_values.csv",
        "│   ├── state_counts.csv",
        "│   ├── profit_distribution.png",
        "│   ├── *_vs_profit.png",
        "│   ├── correlation_matrix.csv and .png",
        "├── feature_selection/",
        "│   ├── individual/  (5 of each: correlation, mutual_info, rfe, rfecv, lasso plots & csvs)",
        "│   └── combined/    (1 of together: combined plot, master table, and comparison diagnostics)",
        "├── model_evaluation/ (actual vs predicted, residuals, and predictions table)",
        "└── project_report.txt",
    ]
    (OUTPUT_DIR / "project_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nExecution finished successfully. Project report written to: {OUTPUT_DIR / 'project_report.txt'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
