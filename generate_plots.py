"""
generate_plots.py
-----------------
Reproduces the exploratory data analysis from the R script
(R/online_news_popularity_analysis.R) using Python, and adds a few
extra diagnostic plots (Q-Q plots) that are useful to include in the
project write-up / README but were not in the original R script.

This was written because the sandbox used to prepare the GitHub repo
did not have R installed. The plots below use the exact same logic
(median-split target, correlation-threshold feature selection, etc.)
as the R script, so the figures are consistent with what `Rscript
R/online_news_popularity_analysis.R` would produce if run locally.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
df = pd.read_csv("data/OnlineNewsPopularity.csv")
df.columns = [c.strip() for c in df.columns]

# Drop non-predictive columns (mirrors the R script)
df = df.drop(columns=["url", "timedelta"])

# Median-split target
threshold = df["shares"].median()
df["Popularity"] = np.where(df["shares"] >= threshold, "Popular", "Unpopular")

# Keep 'shares' around locally for the Q-Q plots, then drop for modeling parity
shares = df["shares"].copy()
df_model = df.drop(columns=["shares"])

# Feature engineering (mirrors the R script)
df_model["content_title_ratio"] = df_model["n_tokens_content"] / (df_model["n_tokens_title"] + 1)
df_model["media_score"] = df_model["num_imgs"] + df_model["num_videos"]

palette = {"Popular": "lightgreen", "Unpopular": "salmon"}

# ---------------------------------------------------------------
# 2. Distribution of Popular vs Unpopular
# ---------------------------------------------------------------
plt.figure(figsize=(6, 5))
order = ["Unpopular", "Popular"]
sns.countplot(data=df_model, x="Popularity", order=order,
              palette=palette, edgecolor="black")
plt.title("Distribution of News Popularity")
plt.xlabel("Class")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.savefig("plots/01_popularity_distribution.png")
plt.close()

# ---------------------------------------------------------------
# 3. Content length vs Popularity (log scale boxplot)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 5))
tmp = df_model.copy()
tmp["n_tokens_content_log"] = tmp["n_tokens_content"] + 1
sns.boxplot(data=tmp, x="Popularity", y="n_tokens_content_log", order=order,
            palette=palette)
plt.yscale("log")
plt.title("Content Length vs Popularity")
plt.xlabel("Class")
plt.ylabel("Word Count (log scale + 1)")
plt.tight_layout()
plt.savefig("plots/02_content_length_vs_popularity.png")
plt.close()

# ---------------------------------------------------------------
# 4. Number of images vs Popularity
# ---------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.boxplot(data=df_model, x="Popularity", y="num_imgs", order=order,
            palette=palette, showfliers=False)
plt.ylim(0, 20)
plt.title("Number of Images vs Popularity")
plt.xlabel("Class")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig("plots/03_num_images_vs_popularity.png")
plt.close()

# ---------------------------------------------------------------
# 5. Correlation matrix of highly-correlated features (|r| > 0.05)
# ---------------------------------------------------------------
popularity_numeric = (df_model["Popularity"] == "Popular").astype(int)
numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

corr_with_target = df_model[numeric_cols].apply(lambda col: col.corr(popularity_numeric))
highly_correlated = corr_with_target[corr_with_target.abs() > 0.05].index.tolist()

corr_matrix = df_model[highly_correlated].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True,
            linewidths=0.4, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Highly Correlated Features (|r| > 0.05 with Popularity)")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig("plots/04_correlation_heatmap.png")
plt.close()

# Bar chart of each feature's correlation with the target (easy to read at a glance)
plt.figure(figsize=(8, 6))
corr_sorted = corr_with_target.loc[highly_correlated].sort_values()
colors = ["salmon" if v < 0 else "lightgreen" for v in corr_sorted.values]
plt.barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor="black")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Popularity (|r| > 0.05)")
plt.xlabel("Correlation coefficient")
plt.tight_layout()
plt.savefig("plots/05_feature_correlation_with_target.png")
plt.close()

# ---------------------------------------------------------------
# 6. Q-Q plots (normality diagnostics) - shares raw vs log-transformed
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

stats.probplot(shares, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot: Raw 'shares'")

log_shares = np.log1p(shares)
stats.probplot(log_shares, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: log(1 + shares)")

plt.tight_layout()
plt.savefig("plots/06_qq_plots_shares.png")
plt.close()

# Q-Q plot for a couple of key predictors as well
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
stats.probplot(df_model["n_tokens_content"], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot: n_tokens_content")

stats.probplot(df_model["global_sentiment_polarity"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: global_sentiment_polarity")

plt.tight_layout()
plt.savefig("plots/07_qq_plots_predictors.png")
plt.close()

# ---------------------------------------------------------------
# 7. Class balance / summary stats saved to text for the README
# ---------------------------------------------------------------
summary_lines = [
    f"Rows: {df_model.shape[0]}, Columns (after drop/engineering): {df_model.shape[1]}",
    f"Median shares (split threshold): {threshold}",
    f"Class balance:\n{df_model['Popularity'].value_counts().to_string()}",
    f"\nFeatures with |correlation| > 0.05 vs Popularity ({len(highly_correlated)} of {len(numeric_cols)}):",
    corr_with_target.loc[highly_correlated].sort_values(ascending=False).to_string(),
]
with open("plots/summary_stats.txt", "w") as f:
    f.write("\n\n".join(summary_lines))

print("Done. Plots written to plots/")
print("\n".join(summary_lines))
