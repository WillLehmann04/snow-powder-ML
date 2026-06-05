"""
EDA — Visualise training_dataset.parquet before Phase 4 modelling.

Produces 6 plot files in data/plots/:
  01_target_distribution.png   — overnight_snow_cm histogram + log-scale
  02_feature_correlations.png  — heatmap of Pearson r with target
  03_scatter_top_features.png  — scatter of top-8 correlated features vs target
  04_powder_by_resort.png      — powder day % and mean snowfall per resort
  05_snowfall_by_month.png     — monthly snowfall distribution (violin)
  06_region_comparison.png     — CDF of overnight snow by region

Run:
  python explore.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, saves files without needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Setup ─────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
TARGET = "overnight_snow_cm"

print("Loading dataset …")
df = pd.read_parquet("data/processed/training_dataset.parquet")
print(f"  {len(df):,} rows, {df.shape[1]} columns\n")

# Numeric feature columns only (exclude identifiers / categoricals)
EXCLUDE = {"date", "resort_id", "season", "region", "overnight_snow_cm",
           "snow_depth_cm", "season_year", "region_code"}
NUM_COLS = [c for c in df.select_dtypes(include="number").columns if c not in EXCLUDE]

# ── 1. Target distribution ────────────────────────────────────────────────────
print("Plot 1: target distribution …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Target Distribution — overnight_snow_cm", fontsize=14, fontweight="bold")

axes[0].hist(df[TARGET], bins=80, color="#4c8cbf", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("overnight_snow_cm")
axes[0].set_ylabel("Count")
axes[0].set_title("Linear scale")

log_vals = df[df[TARGET] > 0][TARGET]
axes[1].hist(np.log1p(log_vals), bins=60, color="#e07b54", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("log(1 + overnight_snow_cm)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log scale (zero-snow days excluded)")

powder_pct = 100 * (df[TARGET] >= 15).mean()
axes[0].axvline(15, color="gold", linewidth=1.5, linestyle="--", label=f"15cm threshold ({powder_pct:.1f}% of days)")
axes[0].legend()

plt.tight_layout()
fig.savefig(PLOTS_DIR / "01_target_distribution.png", dpi=150)
plt.close()

# ── 2. Feature correlations with target ───────────────────────────────────────
print("Plot 2: feature correlations …")
corrs = (
    df[NUM_COLS + [TARGET]]
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)

fig, ax = plt.subplots(figsize=(10, max(8, len(corrs) * 0.28)))
colors = ["#e05c5c" if v < 0 else "#4c8cbf" for v in corrs.values]
ax.barh(corrs.index[::-1], corrs.values[::-1], color=colors[::-1], edgecolor="none")
ax.axvline(0, color="white", linewidth=0.8)
ax.set_xlabel("Pearson r with overnight_snow_cm")
ax.set_title("Feature Correlations with Target", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
plt.tight_layout()
fig.savefig(PLOTS_DIR / "02_feature_correlations.png", dpi=150)
plt.close()

# ── 3. Scatter — top 8 features vs target ────────────────────────────────────
print("Plot 3: scatter top features …")
top8 = corrs.abs().nlargest(8).index.tolist()
sample = df.sample(min(4000, len(df)), random_state=42)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Top-8 Features vs overnight_snow_cm (4k sample)", fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(top8):
    axes[i].scatter(sample[col], sample[TARGET], alpha=0.15, s=8, color="#4c8cbf")
    # trend line
    finite = sample[[col, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) > 10:
        z = np.polyfit(finite[col], finite[TARGET], 1)
        p = np.poly1d(z)
        x_line = np.linspace(finite[col].quantile(0.01), finite[col].quantile(0.99), 100)
        axes[i].plot(x_line, p(x_line), color="#e07b54", linewidth=1.5)
    r = corrs[col]
    axes[i].set_title(f"{col}\nr={r:.3f}", fontsize=9)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel(TARGET if i % 4 == 0 else "", fontsize=8)
    axes[i].tick_params(labelsize=7)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "03_scatter_top_features.png", dpi=150)
plt.close()

# ── 4. Powder days by resort ──────────────────────────────────────────────────
print("Plot 4: powder by resort …")
resort_stats = (
    df.groupby("resort_id")
    .agg(
        powder_pct=(TARGET, lambda x: 100 * (x >= 15).mean()),
        mean_snow=(TARGET, "mean"),
        n=(TARGET, "count"),
    )
    .sort_values("powder_pct", ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Per-Resort Snowfall Stats", fontsize=13, fontweight="bold")

y_pos = range(len(resort_stats))
axes[0].barh(y_pos, resort_stats["powder_pct"], color="#4c8cbf", edgecolor="none")
axes[0].set_yticks(list(y_pos))
axes[0].set_yticklabels(resort_stats.index, fontsize=9)
axes[0].set_xlabel("Powder days ≥15cm (%)")
axes[0].set_title("Powder Day Frequency")
axes[0].axvline(resort_stats["powder_pct"].mean(), color="gold", linewidth=1.2,
                linestyle="--", label=f"avg {resort_stats['powder_pct'].mean():.1f}%")
axes[0].legend(fontsize=8)

axes[1].barh(y_pos, resort_stats["mean_snow"], color="#5bb87a", edgecolor="none")
axes[1].set_yticks(list(y_pos))
axes[1].set_yticklabels(resort_stats.index, fontsize=9)
axes[1].set_xlabel("Mean overnight snowfall (cm)")
axes[1].set_title("Mean Overnight Snowfall")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "04_powder_by_resort.png", dpi=150)
plt.close()

# ── 5. Snowfall by month (violin) ─────────────────────────────────────────────
print("Plot 5: snowfall by month …")
month_order = [11, 12, 1, 2, 3, 4, 5]
month_labels = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"]
plot_data = df[df[TARGET] > 0].copy()   # non-zero days only for violin
plot_data["month_name"] = pd.Categorical(
    plot_data["month"].map(dict(zip(month_order, month_labels))),
    categories=month_labels, ordered=True,
)

fig, ax = plt.subplots(figsize=(13, 6))
sns.violinplot(
    data=plot_data, x="month_name", y=TARGET,
    palette="Blues", cut=0, inner="quartile", ax=ax,
)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("Month")
ax.set_ylabel("overnight_snow_cm (log, non-zero days)")
ax.set_title("Overnight Snowfall Distribution by Month", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "05_snowfall_by_month.png", dpi=150)
plt.close()

# ── 6. Region comparison (CDF) ────────────────────────────────────────────────
print("Plot 6: region CDF …")
region_colors = {"hokkaido": "#4c8cbf", "tohoku": "#e07b54",
                 "niigata": "#5bb87a",  "nagano": "#c06bbd"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Overnight Snowfall by Region", fontsize=13, fontweight="bold")

for region, grp in df.groupby("region"):
    color = region_colors.get(region, "grey")
    vals = np.sort(grp[TARGET].values)
    cdf = np.arange(1, len(vals) + 1) / len(vals)
    axes[0].plot(vals, cdf, label=region, color=color, linewidth=1.8)

axes[0].set_xlim(0, 60)
axes[0].set_xlabel("overnight_snow_cm")
axes[0].set_ylabel("CDF")
axes[0].set_title("CDF (0–60cm range)")
axes[0].legend()

sns.boxplot(
    data=df[df[TARGET] > 0], x="region", y=TARGET,
    palette=region_colors, order=sorted(region_colors),
    ax=axes[1], showfliers=False,
)
axes[1].set_ylabel("overnight_snow_cm (non-zero)")
axes[1].set_title("Box plot (non-zero days, outliers hidden)")
axes[1].set_xlabel("Region")

plt.tight_layout()
fig.savefig(PLOTS_DIR / "06_region_comparison.png", dpi=150)
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
plots = sorted(PLOTS_DIR.glob("*.png"))
print(f"\nSaved {len(plots)} plots to {PLOTS_DIR}/")
for p in plots:
    print(f"  {p.name}")
