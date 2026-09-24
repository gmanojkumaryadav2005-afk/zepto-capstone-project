"""
Module 2 - Analytics, Part A: Profiling, cleaning, and the data story.

Run:
    python 01_eda.py
Produces:
    titanic.csv                  (offline fallback of the raw loaded dataset)
    titanic_clean.csv            (cleaned dataset used downstream)
    charts/*.png                 (>=4 multivariate charts + heatmap + z-score check)
    eda_report.txt                (all printed/interpreted text output)
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)
report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# ---------------------------------------------------------------------------
# Load ONCE. Save an offline fallback immediately.
# ---------------------------------------------------------------------------
df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)  # offline fallback per spec
log("Loaded titanic dataset via sns.load_dataset and cached as titanic.csv")

# ---------------------------------------------------------------------------
# Task 1: profile
# ---------------------------------------------------------------------------
log("\n=== df.info() ===")
buf = []
df.info(buf=type("W", (), {"write": buf.append})())
log("".join(buf))

log("\n=== df.describe() ===")
log(df.describe(include="all").to_string())

log(f"\nshape: {df.shape}")

missing_pct = (df.isna().mean() * 100).round(2)
log("\n=== % missing per column ===")
log(missing_pct[missing_pct > 0].to_string())

# ---------------------------------------------------------------------------
# Task 2: missing-value handling, per column, threshold rule
#   <5% missing  -> drop rows
#   5-30% missing -> impute
#   >30% missing -> explicitly drop column OR encode "missing" category
# ---------------------------------------------------------------------------
log("\n=== Missing-value strategy ===")
clean = df.copy()
for col in missing_pct[missing_pct > 0].index:
    pct = missing_pct[col]
    if pct < 5:
        clean = clean[clean[col].notna()]
        log(f"{col}: {pct}% missing (<5%) -> dropped rows with missing values")
    elif pct <= 30:
        if clean[col].dtype == "object" or str(clean[col].dtype) == "category":
            fill = clean[col].mode(dropna=True)[0]
            clean[col] = clean[col].fillna(fill)
            log(f"{col}: {pct}% missing (5-30%) -> imputed with mode ('{fill}')")
        else:
            fill = clean[col].median()
            clean[col] = clean[col].fillna(fill)
            log(f"{col}: {pct}% missing (5-30%) -> imputed with median ({fill})")
    else:
        # >30% missing and unreliable to impute (e.g. 'deck'): encode as its
        # own "missing" category rather than dropping the column, so the
        # signal "we don't know" is preserved as a feature.
        clean[col] = clean[col].astype("object").fillna("missing")
        log(f"{col}: {pct}% missing (>30%) -> too unreliable to impute; encoded 'missing' as its own category")

# ---------------------------------------------------------------------------
# Task 3: univariate analysis - age & fare
# ---------------------------------------------------------------------------
log("\n=== Univariate: age & fare ===")
for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(clean[col], bins=30)
    axes[0].set_title(f"{col} histogram")
    axes[1].boxplot(clean[col])
    axes[1].set_title(f"{col} boxplot")
    fig.tight_layout()
    fig.savefig(f"charts/univariate_{col}.png")
    plt.close(fig)

    q1, q3 = clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = clean[(clean[col] < lo) | (clean[col] > hi)]
    log(f"{col}: {len(outliers)} IQR outliers (bounds [{lo:.2f}, {hi:.2f}])")

    mean_, median_, mode_ = clean[col].mean(), clean[col].median(), clean[col].mode()[0]
    if col == "fare":
        skew_dir = "right-skewed (mean > median > mode)" if mean_ > median_ > mode_ else \
                   "left-skewed (mean < median < mode)" if mean_ < median_ < mode_ else "roughly symmetric"
        log(f"fare: mean={mean_:.2f}, median={median_:.2f}, mode={mode_:.2f} -> {skew_dir}")

# ---------------------------------------------------------------------------
# Task 4: bivariate analysis + correlation heatmap
# ---------------------------------------------------------------------------
log("\n=== Bivariate: survival rate by sex, pclass, sex+pclass ===")
log("By sex:\n" + clean.groupby("sex")["survived"].mean().to_string())
log("By pclass:\n" + clean.groupby("pclass")["survived"].mean().to_string())
log("By sex & pclass:\n" + clean.groupby(["sex", "pclass"])["survived"].mean().to_string())

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix (6 numeric columns)")
fig.tight_layout()
fig.savefig("charts/correlation_heatmap.png")
plt.close(fig)

corr_pairs = corr.where(~np.eye(len(corr), dtype=bool)).abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[~corr_pairs.index.duplicated()]
top2 = corr_pairs.head(2)
log(f"\nTwo strongest correlations (by |r|):\n{top2.to_string()}")
log(
    "Interpretation: the strongest off-diagonal pairs indicate which raw "
    "features move together most; e.g. fare/pclass and parch/sibsp typically "
    "surface here since fare is driven by cabin class and family-size "
    "features co-occur."
)

# ---------------------------------------------------------------------------
# Task 5: multivariate "data story" - >= 4 charts, each with interpretation
# ---------------------------------------------------------------------------
log("\n=== Multivariate data story (4 charts) ===")

fig, ax = plt.subplots()
sns.barplot(data=clean, x="pclass", y="survived", hue="sex", ax=ax)
ax.set_title("Survival rate by class and sex")
fig.savefig("charts/story_1_class_sex_survival.png")
plt.close(fig)
log(
    "Chart 1 (bar): Survival rate is highest for women in 1st/2nd class and "
    "lowest for men in 3rd class, showing 'women and children first' combined "
    "with class privilege in access to lifeboats."
)

fig, ax = plt.subplots()
sns.boxplot(data=clean, x="survived", y="fare", ax=ax)
ax.set_title("Fare distribution by survival")
fig.savefig("charts/story_2_fare_survival_box.png")
plt.close(fig)
log(
    "Chart 2 (box): Survivors show a higher median fare and a longer upper "
    "tail than non-survivors, consistent with higher-fare (often 1st class) "
    "passengers surviving more often."
)

fig, ax = plt.subplots()
sns.scatterplot(data=clean, x="age", y="fare", hue="survived", alpha=0.6, ax=ax)
ax.set_title("Age vs fare, colored by survival")
fig.savefig("charts/story_3_age_fare_scatter.png")
plt.close(fig)
log(
    "Chart 3 (scatter): Survivors (orange) skew toward higher fares across "
    "most ages, while very young children show relatively high survival even "
    "at low fares, matching the age-based rescue priority."
)

pair_data = clean[["survived", "pclass", "age", "fare"]].dropna()
pp = sns.pairplot(pair_data, hue="survived", diag_kind="hist")
pp.savefig("charts/story_4_pairplot.png")
plt.close("all")
log(
    "Chart 4 (pair-plot): Across pclass/age/fare, the clearest separation "
    "between survived/not-survived classes appears along pclass and fare, "
    "while age alone separates the two groups less cleanly."
)

# ---------------------------------------------------------------------------
# Task 6: exploratory z-score standardization sanity check (age, fare)
# ---------------------------------------------------------------------------
log("\n=== Exploratory z-score check (age, fare) ===")
before = clean[["age", "fare"]].agg(["mean", "std"])
z = (clean[["age", "fare"]] - clean[["age", "fare"]].mean()) / clean[["age", "fare"]].std()
after = z.agg(["mean", "std"])
log("Before:\n" + before.to_string())
log("After z-score transform (should be ~mean 0, std 1):\n" + after.to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(clean["age"], bins=30, alpha=0.5, label="age (raw)")
axes[0].hist(z["age"], bins=30, alpha=0.5, label="age (z)")
axes[0].legend()
axes[1].hist(clean["fare"], bins=30, alpha=0.5, label="fare (raw)")
axes[1].hist(z["fare"], bins=30, alpha=0.5, label="fare (z)")
axes[1].legend()
fig.tight_layout()
fig.savefig("charts/zscore_before_after.png")
plt.close(fig)
log("(This is an EDA-stage sanity check only; it does not feed the modeling pipeline in 02_modeling.py.)")

# ---------------------------------------------------------------------------
# Save cleaned data for the modeling stage to reuse
# ---------------------------------------------------------------------------
clean.to_csv("titanic_clean.csv", index=False)

with open("eda_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\nDone. See titanic.csv, titanic_clean.csv, charts/, eda_report.txt")
