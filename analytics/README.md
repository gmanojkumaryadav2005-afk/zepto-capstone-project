# Module 2 — Analytics Pipeline

## Run
```bash
pip install -r requirements.txt
python 01_eda.py        # loads titanic once, cleans it, saves titanic.csv + titanic_clean.csv, EDA charts
python 02_modeling.py   # reads titanic_clean.csv, trains/evaluates/tunes models, saves best_pipeline.joblib
```
`01_eda.py` needs internet the first time it runs (Seaborn fetches and caches
the Titanic dataset). It also writes `titanic.csv` as an offline fallback, so
`02_modeling.py` never needs the network.

## Design decisions
- **One cohesive pipeline**: the raw dataset is loaded exactly once, in
  `01_eda.py`; `02_modeling.py` starts from the cleaned CSV that produces —
  never a second `sns.load_dataset` call.
- **Missing-value strategy**: per-column threshold rule — <5% missing rows
  are dropped, 5–30% is imputed (median for numeric, mode for categorical),
  and >30% missing columns (e.g. `deck`) are encoded as an explicit
  "missing" category rather than dropped, since absence of a cabin record is
  itself informative for survival. Exact percentages and the strategy
  applied to each column are printed and logged to `eda_report.txt`.
- **Correlation matrix**: restricted to the six specified numeric columns
  (`survived, pclass, age, sibsp, parch, fare`); `adult_male`/`alone` are
  excluded as derived/redundant flags per the spec.
- **Train/test split**: stratified on `survived` before any preprocessing,
  justified by the observed class imbalance.
- **Preprocessing**: a `ColumnTransformer` (median/mode imputation, one-hot
  encoding, `StandardScaler`) wrapped in a `Pipeline`, fit only on the
  training split and applied in transform-only mode to the test split —
  enforced structurally rather than by hand.
- **Models**: Logistic Regression, Decision Tree, Random Forest, evaluated
  with confusion matrix / accuracy / precision / recall / F1 / ROC-AUC,
  compared side by side.
- **Imbalance handling**: baseline vs. `class_weight='balanced'` vs. SMOTE
  (applied to the training fold only, after the split, to avoid leakage).
- **Tuning**: `GridSearchCV` over `n_estimators`, `max_depth`,
  `max_features` for Random Forest, constructed with `oob_score=True` at
  construction time so the OOB score is available to report.
- **Regression side-task**: predicts `fare` from the other features with
  multivariate linear regression; reports MAE/RMSE/R²/Adjusted R² and a
  residual plot, with a written heteroscedasticity conclusion.
- **Final table**: classification and regression metrics are presented as
  two separate metric groups (not implied to be on one shared scale),
  followed by a written recommendation of which classifier to deploy.
- **Saved artifact**: the complete fitted pipeline (preprocessing + final
  estimator) is saved with `joblib.dump`, then reloaded and confirmed to
  predict correctly on raw, unprocessed input.

All printed output and written interpretations are captured in
`eda_report.txt` and `modeling_report.txt`; charts are saved under
`charts/`.
