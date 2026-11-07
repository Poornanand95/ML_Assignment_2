# ShopSense — Predicting Purchase Intent of Online Shoppers

**ML Assignment 2 · M.Tech (AIML), BITS Pilani WILP**

- **GitHub repository:** `<GITHUB_REPO_LINK>`
- **Live Streamlit app:** `<STREAMLIT_APP_LINK>`

---

## a. Problem statement

An e-commerce site sees thousands of browsing sessions a day, but only a small share of them end in a
purchase. The goal is to build a **binary classifier** that predicts, from the behaviour recorded during a
session, whether the visitor will generate revenue (`Revenue = 1`) or leave without buying (`Revenue = 0`).

Five classification algorithms — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive
Bayes and Support Vector Machine — are trained on the same data, evaluated on the same held-out test set with
six metrics, and served through an interactive Streamlit app that loads the saved models and evaluates them on
an uploaded test CSV.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (ID 468) — <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| Instances | 12,330 sessions (one row per session, one year of data, each session from a different user) |
| Features | 17 input features + 1 target |
| Target | `Revenue` — TRUE/FALSE (1/0) |
| Class balance | 10,422 no-purchase (84.5 %) vs 1,908 purchase (15.5 %) — **imbalanced** |
| Missing values | None |

**Features**

| Type | Columns | Meaning |
|---|---|---|
| Numeric (10) | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` | number of pages of each type visited and total time (s) spent on them |
| | `BounceRates`, `ExitRates`, `PageValues` | Google Analytics metrics of the pages visited |
| | `SpecialDay` | closeness of the visit date to a special day (e.g. Mother's Day, Valentine's Day), 0–1 |
| Categorical (6) | `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` | when / how / from where the visitor arrived; returning vs new visitor |
| Binary (1) | `Weekend` | whether the session was on a weekend |

**Pre-processing** (built into every saved scikit-learn `Pipeline`, so the app needs only the raw columns)

- Numeric columns: `log1p` transform (page counts, durations and `PageValues` are very right-skewed) followed by `StandardScaler`.
- Categorical columns: `OneHotEncoder(handle_unknown="ignore")`.
- `Weekend`: passed through as 0/1.
- 80 / 20 **stratified** train–test split (`random_state = 2611`): 9,864 training rows, 2,466 test rows (`test_data.csv`).
- Hyper-parameters tuned with `GridSearchCV`, 5-fold stratified CV, scoring = **F1** (accuracy alone is misleading at 15 % positives). Logistic Regression, Decision Tree and SVM use `class_weight="balanced"`.

## c. GitHub repository link

`<GITHUB_REPO_LINK>`

```
.
├── app.py                 # Streamlit app (upload CSV → pick model → metrics, confusion matrix, report)
├── train_models.py        # trains, tunes and evaluates all five models; saves them to model/
├── shopper_utils.py       # shared column lists, cleaning and metric helpers
├── requirements.txt
├── README.md
├── test_data.csv          # held-out 20 % test split used for all reported results
├── data/
│   ├── online_shoppers_intention.csv   # original UCI file
│   └── train_data.csv                  # 80 % training split
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metrics.json       # test metrics + best hyper-parameters
└── .streamlit/config.toml
```

**Reproduce**

```bash
pip install -r requirements.txt
python train_models.py      # re-creates the split, models and metrics.json
streamlit run app.py
```

## d. Models used

Best hyper-parameters (from 5-fold CV):

| Model | Best hyper-parameters | CV F1 |
|---|---|---|
| Logistic Regression | C = 0.01, class_weight = balanced | 0.6639 |
| Decision Tree | max_depth = 4, min_samples_leaf = 30, class_weight = balanced | 0.6364 |
| KNN | n_neighbors = 31, weights = distance | 0.6205 |
| Naive Bayes (Gaussian) | var_smoothing = 0.1 | 0.5151 |
| SVM (RBF kernel) | C = 0.5, gamma = 0.01, class_weight = balanced | 0.6664 |

### Comparison table (held-out test set, 2,466 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8710 | 0.9171 | 0.5576 | 0.8115 | 0.6610 | 0.6004 |
| Decision Tree | 0.8617 | 0.9144 | 0.5336 | 0.8534 | 0.6566 | 0.6006 |
| KNN | **0.8974** | 0.9205 | **0.7115** | 0.5681 | 0.6317 | 0.5779 |
| Naive Bayes | 0.7571 | 0.8777 | 0.3766 | **0.8665** | 0.5250 | 0.4559 |
| Support Vector Machine | 0.8763 | **0.9211** | 0.5701 | 0.8194 | **0.6724** | **0.6141** |

*AUC for the SVM is computed from its decision-function scores; for the other models from predicted probabilities.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong, simple baseline. After log-scaling and one-hot encoding the problem is close to linearly separable — `PageValues` alone carries most of the signal — (on its own it gives a test AUC of 0.87) — so LR reaches AUC 0.917 and MCC 0.600. Heavy regularisation (C = 0.01) worked best, meaning the many one-hot columns add little. Balanced weights push recall to 0.81 at the cost of precision (0.56). |
| Decision Tree | A shallow tree (depth 4, ≥ 30 samples per leaf) generalised best; deeper trees over-fitted in CV. It has the second-highest recall (0.85) and an MCC (0.601) equal to LR, but the lowest precision of the four good models (0.53) — its few coarse splits label many borderline sessions as buyers. Easy to interpret: the first split is on `PageValues`. |
| KNN | Highest accuracy (0.897) and precision (0.71), but lowest recall among the good models (0.57): with no class weighting the 85 % majority dominates the neighbourhoods, so it predicts "purchase" only when it is very confident. Its high accuracy is partly a result of the imbalance — MCC (0.578) shows it is actually slightly weaker than LR/DT/SVM. Needs k = 31 with distance weighting to smooth noisy neighbourhoods. |
| Naive Bayes | Weakest model (accuracy 0.757, MCC 0.456). The independence assumption is clearly violated (page counts and their durations are highly correlated, and the one-hot columns are not Gaussian), so probabilities are poorly calibrated and it over-predicts purchases: highest recall (0.87) but precision of only 0.38. Still useful as a very fast, high-recall screen. |
| Support Vector Machine | Best overall: highest AUC (0.921), F1 (0.672) and MCC (0.614) with a good recall/precision balance (0.82 / 0.57). The RBF kernel can model mild non-linear feature interactions that LR's single linear boundary cannot, and balanced class weights handle the imbalance. Downsides: slowest to train (~18 s for the grid vs < 5 s for the others) and no native probabilities. |
| **Overall winner for this dataset** | **Support Vector Machine** — it ranks first on the three imbalance-robust metrics (AUC, F1, MCC). Logistic Regression is a close second and a good choice if interpretability or training speed matters more. If the business only wants to act on high-confidence buyers (precision over recall), KNN would be preferred. |

## Streamlit app features

- **CSV upload** of test data (plus a button to download `test_data.csv`, and a toggle to use the bundled copy).
- **Model selection dropdown** for the five saved models.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC shown as metric cards.
- **Confusion matrix** (interactive heatmap), **ROC curve** and **classification report**.
- **Compare all models** tab: all five models scored on the uploaded file, with a highlighted bar chart for any metric.
- **Predictions** tab: per-row predictions, misclassified-only filter and CSV download.
- Models are only loaded with `joblib` — nothing is re-trained on the uploaded data.
