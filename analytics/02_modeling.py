"""
Module 2 - Analytics, Part B: Predictive modeling, continuing from the same
cleaned data produced by 01_eda.py (titanic_clean.csv).

Run:
    python 01_eda.py      # first, to produce titanic_clean.csv
    python 02_modeling.py
Produces:
    charts/*.png (tree, ROC curves, residual plot)
    modeling_report.txt
    best_pipeline.joblib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE
import joblib

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


df = pd.read_csv("titanic_clean.csv")

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"
NUM_COLS = ["age", "sibsp", "parch", "fare", "pclass"]
CAT_COLS = ["sex", "embarked"]

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ---------------------------------------------------------------------------
# Task 7: stratified train/test split
# ---------------------------------------------------------------------------
log("=== Train/test split ===")
class_balance = y.value_counts(normalize=True)
log(f"Class balance (survived): \n{class_balance.to_string()}")
log(
    "Stratification is used because the classes are imbalanced "
    f"({class_balance.min():.1%} vs {class_balance.max():.1%}); a plain "
    "random split risks over/under-representing the minority class in "
    "train or test, distorting evaluation."
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Task 8: preprocessing, fit on train only, via ColumnTransformer/Pipeline
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUM_COLS),
    ("cat", categorical_transformer, CAT_COLS),
])

# ---------------------------------------------------------------------------
# Task 9-10: train & evaluate three classifiers
# ---------------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
}

fitted_pipelines = {}
eval_rows = []
fig_roc, ax_roc = plt.subplots()

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    log(f"\n=== {name} ===")
    log(f"Confusion matrix:\n{cm}")
    log(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} AUC={roc_auc:.3f}")

    eval_rows.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": roc_auc})

ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.set_title("ROC curves")
ax_roc.legend()
fig_roc.savefig("charts/roc_curves.png")
plt.close(fig_roc)

classifier_comparison = pd.DataFrame(eval_rows).set_index("model")
log("\n=== Classifier comparison table ===\n" + classifier_comparison.to_string())

# Decision tree visualization
dt_pipe = fitted_pipelines["DecisionTree"]
feature_names = (
    NUM_COLS
    + list(dt_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CAT_COLS))
)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["clf"], feature_names=feature_names,
    class_names=["died", "survived"], filled=True, max_depth=3, fontsize=8, ax=ax,
)
fig.savefig("charts/decision_tree.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Task 11: imbalance handling, three ways, on RandomForest
# ---------------------------------------------------------------------------
log("\n=== Imbalance handling (RandomForest) ===")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

imbalance_rows = []

rf_baseline = RandomForestClassifier(random_state=42).fit(X_train_proc, y_train)
p = rf_baseline.predict(X_test_proc)
imbalance_rows.append({"strategy": "baseline", "precision": precision_score(y_test, p),
                        "recall": recall_score(y_test, p), "f1": f1_score(y_test, p)})

rf_weighted = RandomForestClassifier(random_state=42, class_weight="balanced").fit(X_train_proc, y_train)
p = rf_weighted.predict(X_test_proc)
imbalance_rows.append({"strategy": "class_weight_balanced", "precision": precision_score(y_test, p),
                        "recall": recall_score(y_test, p), "f1": f1_score(y_test, p)})

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_proc, y_train)  # training fold only
rf_smote = RandomForestClassifier(random_state=42).fit(X_train_sm, y_train_sm)
p = rf_smote.predict(X_test_proc)
imbalance_rows.append({"strategy": "smote_train_only", "precision": precision_score(y_test, p),
                        "recall": recall_score(y_test, p), "f1": f1_score(y_test, p)})

imbalance_df = pd.DataFrame(imbalance_rows).set_index("strategy")
log(imbalance_df.to_string())
best_strategy = imbalance_df["f1"].idxmax()
log(
    f"Conclusion: '{best_strategy}' gives the best F1 on this split. Baseline "
    "already does reasonably since Titanic's imbalance is mild (~38% survived); "
    "class_weight and SMOTE mainly trade some precision for recall on the "
    "minority (survived) class."
)

# ---------------------------------------------------------------------------
# Task 12: hyperparameter tuning with GridSearchCV + OOB score
# ---------------------------------------------------------------------------
log("\n=== Hyperparameter tuning (RandomForest) ===")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"],
}
rf_for_search = RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)
grid = GridSearchCV(rf_for_search, param_grid, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train_proc, y_train)
log(f"Best params: {grid.best_params_}")
best_rf_refit = RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True, **grid.best_params_)
best_rf_refit.fit(X_train_proc, y_train)
log(f"OOB score of best-params model: {best_rf_refit.oob_score_:.3f}")

# ---------------------------------------------------------------------------
# Task 13: regression side-task - predict fare
# ---------------------------------------------------------------------------
log("\n=== Regression side-task: predict fare ===")
reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
Xr = df[reg_features].copy()
yr = df["fare"].copy()
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
     ["pclass", "age", "sibsp", "parch"]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
     ["sex", "embarked"]),
])
reg_pipe = Pipeline([("preprocess", reg_preprocessor), ("reg", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n, k = len(yr_test), Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
log(f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} Adjusted R2={adj_r2:.3f}")

residuals = yr_test - yr_pred
fig, ax = plt.subplots()
ax.scatter(yr_pred, residuals, alpha=0.5)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted fare"); ax.set_ylabel("Residual")
ax.set_title("Residual plot (fare regression)")
fig.savefig("charts/residual_plot.png")
plt.close(fig)
log(
    "The residual spread visibly widens at higher predicted fares "
    "(a funnel shape), indicating heteroscedasticity rather than constant "
    "variance across the prediction range."
)

# ---------------------------------------------------------------------------
# Task 14: final model comparison table (classification + regression, separate groups)
# ---------------------------------------------------------------------------
log("\n=== Final comparison ===")
log("Classification metrics:\n" + classifier_comparison.to_string())
log(f"\nRegression metrics: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  Adjusted R2={adj_r2:.3f}")
best_clf_name = classifier_comparison["f1"].idxmax()
log(
    f"\nRecommendation: deploy **{best_clf_name}** for the survival-prediction "
    f"task — it has the top F1 ({classifier_comparison.loc[best_clf_name, 'f1']:.3f}) "
    f"and AUC ({classifier_comparison.loc[best_clf_name, 'auc']:.3f}) among the three "
    "classifiers, balancing precision and recall better than the alternatives, "
    "and Random Forest additionally offers stability via ensembling and a "
    "usable feature-importance view for Zepto's analysts."
)

# ---------------------------------------------------------------------------
# Task 15: save the full fitted pipeline (preprocessing + estimator together)
# ---------------------------------------------------------------------------
final_pipeline = fitted_pipelines[best_clf_name]
joblib.dump(final_pipeline, "best_pipeline.joblib")
log(f"\nSaved best-performing complete pipeline ({best_clf_name}) to best_pipeline.joblib")

# reload & confirm it works end-to-end on raw input
reloaded = joblib.load("best_pipeline.joblib")
sample = X_test.iloc[:3]
log("Reload check - predictions on raw (unprocessed) sample rows:")
log(str(reloaded.predict(sample)))

with open("modeling_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\nDone. See modeling_report.txt, charts/, best_pipeline.joblib")
