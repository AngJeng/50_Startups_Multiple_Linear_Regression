<img width="1624" height="999" alt="截圖 2026-10-14 中午12 03 39" src="https://github.com/user-attachments/assets/7f6df446-f204-40b8-b4c0-954e6a9fbcbe" />
<img width="1624" height="999" alt="截圖 2026-10-14 中午12 03 31" src="https://github.com/user-attachments/assets/9e1b109e-32ec-45e5-badd-fdf76febf9b9" />

# 50 Startups — Multiple Linear Regression & Feature Selection (CRISP-DM)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/AngJeng/50_Startups_Multiple_Linear_Regression)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Dashboard-000000?logo=vercel)](https://webdashboard-dusky.vercel.app)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A complete, production-ready machine learning project implementing **Multiple Linear Regression** according to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology. The project systematically analyzes, evaluates, and compares **five feature selection algorithms** across feature subset sizes $k = 1$ to $5$ to eliminate noise, combat multicollinearity, and optimize generalization performance.

🌐 **Live Public Web Dashboard**: [https://webdashboard-dusky.vercel.app](https://webdashboard-dusky.vercel.app)  
📦 **GitHub Repository**: [https://github.com/AngJeng/50_Startups_Multiple_Linear_Regression](https://github.com/AngJeng/50_Startups_Multiple_Linear_Regression)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [Dataset Description](#dataset-description)
4. [CRISP-DM Workflow](#crisp-dm-workflow)
5. [Feature Selection Algorithms Evaluated](#feature-selection-algorithms-evaluated)
6. [Empirical Results & Benchmarks](#empirical-results--benchmarks)
7. [Web Dashboard & Interactive Profit Predictor](#web-dashboard--interactive-profit-predictor)
8. [Installation & Setup](#installation--setup)
9. [Running the Python Pipeline](#running-the-python-pipeline)
10. [Local Dashboard Execution](#local-dashboard-execution)
11. [Vercel Deployment Architecture](#vercel-deployment-architecture)

---

## 1. Project Overview

The objective of this project is to predict startup **Profit** based on operational expenditures (`R&D Spend`, `Administration`, `Marketing Spend`) and geographic location (`State`). In addition to standard regression modeling, this project provides an extensive comparative study of five distinct feature selection techniques (Filter, Wrapper, and Embedded methods) to answer a fundamental data science question: **Does using all available features always yield the best predictive accuracy, or does feature selection produce a superior, more parsimonious model?**

Key Takeaway: The empirical results demonstrate that a **2-feature model** (`R&D Spend` + `Marketing Spend`) consistently achieves lower root mean squared error (RMSE: **$8,198.80**) and higher out-of-sample explanatory power ($R^2$: **0.9474**) than the full 5-feature baseline ($R^2$: **0.9347**, RMSE: **$9,137.99**).

---

## 2. Project Architecture

The repository cleanly separates the reproducible Python machine learning pipeline from the static web dashboard:

```text
50_Startups_Multiple_Linear_Regression/
├── data/
│   └── 50_Startups.csv                        # Raw dataset (50 company observations)
├── models/
│   ├── startup_profit_model.joblib            # Serialized optimal 2-feature pipeline
│   └── baseline_model.joblib                  # Serialized full 5-feature baseline pipeline
├── outputs/                                   # Cleanly organized analytical outputs
│   ├── eda/                                   # Exploratory Data Analysis outputs
│   │   ├── data_summary.csv                   # Descriptive statistics
│   │   ├── missing_values.csv                 # Missing value audit
│   │   ├── state_counts.csv                   # State category counts
│   │   ├── profit_distribution.png            # Target distribution histogram
│   │   ├── *_vs_profit.png                    # Feature scatter plots
│   │   └── correlation_matrix.csv & .png      # Feature & target correlation heatmap
│   ├── feature_selection/
│   │   ├── individual/                        # 5 of Each (k=1..5 dual plots & tables)
│   │   │   ├── correlation_k_features.png & .csv
│   │   │   ├── mutual_information_k_features.png & .csv
│   │   │   ├── rfe_k_features.png & .csv
│   │   │   ├── rfecv_k_features.png & .csv
│   │   │   └── lasso_k_features.png & .csv
│   │   └── combined/                          # 1 of Together (master plots & tables)
│   │       ├── combined_algorithms_k_features.png
│   │       ├── all_algorithms_k_features_comparison.csv
│   │       ├── feature_selection_comparison.csv
│   │       ├── feature_importance_ranking.csv
│   │       ├── rfecv_curve.png
│   │       └── lasso_coefficients.png
│   ├── model_evaluation/                      # Final model validation
│   │   ├── actual_vs_predicted.png            # Actual vs predicted scatter plot
│   │   ├── residual_analysis.png              # Residual distribution & scatter
│   │   ├── test_predictions.csv               # Test set actual vs predicted values
│   │   └── model_coefficients.csv             # Fitted regression coefficients
│   └── project_report.txt                     # Text summary report
├── web_dashboard/                             # Static Web Dashboard for Vercel
│   ├── assets/                                # Static images and raw dataset
│   │   ├── eda/                               # EDA figures
│   │   ├── fs_combined/                       # Combined benchmark figures
│   │   ├── fs_individual/                     # Individual algorithm curves
│   │   ├── eval/                              # Evaluation figures
│   │   └── 50_Startups.csv                    # Downloadable dataset
│   ├── app.js                                 # Client-side routing & interactive logic
│   ├── data.js                                # Embedded JSON datasets and metrics
│   ├── index.html                             # Semantic HTML structure & accessible layout
│   ├── style.css                              # Glassmorphic responsive dark mode styling
│   └── vercel.json                            # Vercel deployment routing configuration
├── 50_Startups_CRISP_DM.ipynb                 # Interactive Jupyter Notebook companion
├── main.py                                    # Full end-to-end Python pipeline
├── predict_new_startup.py                     # CLI script to predict profit for new startups
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation
```

---

## 3. Dataset Description

The dataset consists of financial performance metrics for **50 early-stage ventures** across 3 US states:

| Column Name | Data Type | Units / Range | Description |
|---|---|---|---|
| `R&D Spend` | Continuous Float | $0.00 – $165,349.20 | Annual investment in research and development |
| `Administration` | Continuous Float | $51,283.14 – $182,645.56 | General, administrative, and overhead expenditures |
| `Marketing Spend` | Continuous Float | $0.00 – $471,784.10 | Advertising, customer acquisition, and brand spend |
| `State` | Categorical String | New York, California, Florida | Operational headquarters jurisdiction |
| **`Profit`** | **Continuous Float** | **$14,681.40 – $192,261.83** | **Annual net profit (Target Variable)** |

---

## 4. CRISP-DM Workflow

The project adheres strictly to the six phases of the Cross-Industry Standard Process for Data Mining:

1. **Business Understanding**: Formulate the prediction of venture profitability and establish whether spending efficiency or location determines success.
2. **Data Understanding**: Examine feature distributions, skewness, correlation matrices, and verify that there are zero missing records or exact duplicates.
3. **Data Preparation**: Partition data (80% train / 20% test). Standardize numeric variables using `StandardScaler` to avoid coefficient scale distortion in regularized models and one-hot encode `State` using `drop='first'` to prevent the dummy variable trap.
4. **Modeling**: Construct end-to-end scikit-learn `Pipeline` architectures evaluating five feature selection paradigms alongside ordinary least squares (OLS) linear regression.
5. **Evaluation**: Benchmark all candidate feature subsets across $k = 1$ to $5$ on test RMSE, test $R^2$, and 5-fold cross-validation.
6. **Deployment**: Serialize the highest-performing pipeline to `startup_profit_model.joblib`, implement inference scripts, and deploy the interactive analytical dashboard live on Vercel.

---

## 5. Feature Selection Algorithms Evaluated

To prevent data leakage, all feature selection algorithms are fitted strictly on the training partition:

1. **Correlation Selection (Filter)**: Ranks features descending by absolute Pearson correlation $|r(X_j, y)|$ with target `Profit`.
2. **Mutual Information Selection (Filter)**: Employs `mutual_info_regression` based on entropy and information theory to detect linear and non-linear associations.
3. **Recursive Feature Elimination / RFE (Wrapper)**: Recursively fits linear models, pruning the feature with the lowest coefficient magnitude until the target size $k$ is reached.
4. **RFECV (Cross-Validated Wrapper)**: Evaluates recursive feature elimination inside a 5-fold cross-validation loop to discover the optimal number of features that minimizes cross-validation error.
5. **LASSO Path Selection (Embedded)**: Tracks the L1 regularization shrinkage path ($\lambda \sum |\beta_j|$) as $\alpha$ decreases, observing the entry order of predictors into the active set.

---

## 6. Empirical Results & Benchmarks

### 6.1 Results Across Feature Sizes ($k = 1$ to $5$) for All 5 Algorithms

| Algorithm | Features ($k$) | Selected Feature Subset | Test RMSE ($) | Test $R^2$ Score |
|---|:---:|---|:---:|:---:|
| **Correlation** | 1 | `[R&D Spend]` | $8,274.87 | 0.946459 |
| **Correlation** | **2** | `[R&D Spend, Marketing Spend]` | **$8,198.80** | **0.947439** |
| **Correlation** | 3 | `[R&D Spend, Marketing Spend, Administration]` | $8,803.78 | 0.939396 |
| **Correlation** | 4 | `[R&D Spend, Marketing Spend, Administration, State_New York]` | $8,997.20 | 0.936703 |
| **Correlation** | 5 | `[R&D Spend, Marketing Spend, Administration, State_New York, State_Florida]` | $9,137.99 | 0.934707 |
| **Mutual Information** | 1 | `[R&D Spend]` | $8,274.87 | 0.946459 |
| **Mutual Information** | **2** | `[R&D Spend, Marketing Spend]` | **$8,198.80** | **0.947439** |
| **Mutual Information** | 3 | `[R&D Spend, Marketing Spend, Administration]` | $8,803.78 | 0.939396 |
| **Mutual Information** | 4 | `[R&D Spend, Marketing Spend, Administration, State_Florida]` | $9,068.54 | 0.935696 |
| **Mutual Information** | 5 | `[R&D Spend, Marketing Spend, Administration, State_Florida, State_New York]` | $9,137.99 | 0.934707 |
| **RFE** | 1 | `[R&D Spend]` | $8,274.87 | 0.946459 |
| **RFE** | **2** | `[R&D Spend, Marketing Spend]` | **$8,198.80** | **0.947439** |
| **RFE** | 3 | `[R&D Spend, Marketing Spend, State_Florida]` | $8,376.45 | 0.945136 |
| **RFE** | 4 | `[R&D Spend, Administration, Marketing Spend, State_Florida]` | $9,068.54 | 0.935696 |
| **RFE** | 5 | `[R&D Spend, Administration, Marketing Spend, State_Florida, State_New York]` | $9,137.99 | 0.934707 |
| **RFECV** | 1 | `[R&D Spend]` | $8,274.87 | 0.946459 |
| **RFECV** | **2** | `[R&D Spend, Marketing Spend]` | **$8,198.80** | **0.947439** |
| **RFECV** | 3 | `[R&D Spend, Marketing Spend, State_Florida]` | $8,376.45 | 0.945136 |
| **RFECV** | 4 | `[R&D Spend, Marketing Spend, State_Florida, Administration]` | $9,068.54 | 0.935696 |
| **RFECV** | 5 | `[R&D Spend, Marketing Spend, State_Florida, Administration, State_New York]` | $9,137.99 | 0.934707 |
| **LASSO Path** | 1 | `[R&D Spend]` | $8,274.87 | 0.946459 |
| **LASSO Path** | 2 | `[R&D Spend, State_New York]` | $8,273.83 | 0.946472 |
| **LASSO Path** | 3 | `[R&D Spend, State_New York, State_Florida]` | $8,219.81 | 0.947169 |
| **LASSO Path** | 4 | `[R&D Spend, State_New York, State_Florida, Administration]` | $8,178.37 | 0.947700 |
| **LASSO Path** | 5 | `[R&D Spend, State_New York, State_Florida, Administration, Marketing Spend]` | $9,137.99 | 0.934707 |

### 6.2 Key Statistical Findings
1. **R&D Dominance**: A univariate regression with `R&D Spend` alone achieves $R^2 = 0.9465$, confirming that research and product innovation accounts for over 94.6% of profit variance.
2. **Synergy of R&D and Marketing**: Adding `Marketing Spend` ($k=2$) reduces error to its global minimum of **$8,198.80** and peaks $R^2$ at **0.9474**.
3. **The Penalty of Redundancy**: Adding `Administration` and `State` dummy variables increases parameter estimation variance and causes model degradation (RMSE increases by **+$939.19** to **$9,137.99**).
4. **LASSO Regularization Confirmation**: Standardized LASSO shrivels the coefficients of geographic state variables strictly to zero, proving location has zero marginal utility once operational investments are controlled.

---

## 7. Web Dashboard & Interactive Profit Predictor

The `web_dashboard/` directory contains an interactive, dependency-free web application deployed to Vercel:
- **Feature Selection Explorer**: Dynamically toggle between the combined chart (all 5 algorithms) and individual algorithm curves with synchronized data tables.
- **Live Interactive Profit Predictor**: Adjust financial inputs via sliders or numerical fields to compute real-time expected profit using the production standardized regression formula:
  $$\text{Profit} = \$109,446.45 + 36,426.11 \cdot z_{\text{RD}} + 4,021.92 \cdot z_{\text{Mkt}}$$
- **Dataset & EDA Explorer**: Instant search, sorting, and CSV download of the complete 50 Startups dataset.
- **Model Diagnostics**: High-resolution residual homoscedasticity scatter plots, actual vs. predicted evaluation, and RFECV validation curves.

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Cloning the Repository
```bash
git clone https://github.com/AngJeng/50_Startups_Multiple_Linear_Regression.git
cd 50_Startups_Multiple_Linear_Regression
```

### Virtual Environment Setup
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Running the Python Pipeline

Execute the end-to-end CRISP-DM training, feature selection, and evaluation pipeline:
```bash
python main.py
```
This script will:
1. Load and validate `data/50_Startups.csv`.
2. Generate all exploratory figures into `outputs/eda/`.
3. Compute feature selection across $k=1..5$ for all 5 algorithms and save figures to `outputs/feature_selection/`.
4. Train and serialize the optimal pipeline to `models/startup_profit_model.joblib`.
5. Write the comprehensive report to `outputs/project_report.txt`.

### Predict for a New Startup (CLI)
```bash
python predict_new_startup.py
```
Example Output:
```text
New Startup Profile:
 R&D Spend  Administration  Marketing Spend    State
    100000          120000           250000 New York

Predicted Profit (Best Feature-Selected Model): $133,352.33
Predicted Profit (Baseline All-Features Model):  $133,698.91
```

---

## 10. Local Dashboard Execution

The web dashboard is built using standard HTML5, CSS3, and JavaScript and requires no complex compilation step:

```bash
# Option A: Python Built-in Server
python3 -m http.server 3000 --directory web_dashboard

# Option B: Node.js Serve
npx serve web_dashboard
```
Open [http://localhost:3000](http://localhost:3000) in any modern web browser.

---

## 11. Vercel Deployment Architecture

The dashboard is configured for automatic, zero-downtime deployment on Vercel:
- **Root Directory**: `web_dashboard`
- **Configuration**: [web_dashboard/vercel.json](web_dashboard/vercel.json)
- **Live Deployment**: [https://webdashboard-dusky.vercel.app](https://webdashboard-dusky.vercel.app)

Whenever updates are pushed to the `main` branch of this GitHub repository, Vercel automatically builds and promotes the new deployment.

---

## License
This project is open-source and available under the [MIT License](LICENSE).
