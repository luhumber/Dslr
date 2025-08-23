import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DATA_PATH = 'data/train/dataset_clean.csv'
LABELS_PATH = 'data/train/labels.csv'

data = pd.read_csv(DATA_PATH)
labels = pd.read_csv(LABELS_PATH)

houses = labels['label'].unique()
colors = {
	'Gryffindor': '#E74C3C',
	'Hufflepuff': '#F39C12',
	'Ravenclaw': '#3498DB',
	'Slytherin': '#27AE60'
}

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

def anova_fscore(x: np.ndarray, y: pd.Series) -> float:
	"""One-way ANOVA F-score entre maisons pour une feature x."""
	mask = np.isfinite(x)
	x = x[mask]
	y = y[mask]
	
	groups = [x[y == h] for h in houses]
	sizes = [len(g) for g in groups]
	
	if min(sizes) == 0 or np.nanstd(x) == 0:
		return 0.0
	overall_mean = np.nanmean(x)
	
	ss_between = sum(n_k * (np.nanmean(g) - overall_mean) ** 2 for g, n_k in zip(groups, sizes))
	ss_within = sum(np.nansum((g - np.nanmean(g)) ** 2) for g in groups)
	k = len(groups)
	n = sum(sizes)
	
	if k <= 1 or n <= k or ss_within == 0:
		return 0.0
	
	ms_between = ss_between / (k - 1)
	ms_within = ss_within / (n - k)
	return float(ms_between / ms_within) if ms_within > 0 else 0.0

fscores = {col: anova_fscore(data[col].to_numpy(), labels['label']) for col in numeric_cols}
ranked = sorted(fscores.items(), key=lambda kv: kv[1], reverse=True)

print("Top 10 features by ANOVA F-score (descending):")
for i, (feat, f) in enumerate(ranked[:10], start=1):
	print(f"{i:2d}. {feat}: F = {f:.3f}")

top_k = 5
top_features = [feat for feat, _ in ranked[:top_k]]

def deduplicate_by_corr(features: list[str], threshold: float = 0.9) -> list[str]:
	kept = []
	corr = data[features].corr().abs()
	for f in features:
		if all(corr.loc[f, k] < threshold for k in kept):
			kept.append(f)
	return kept

recommended = deduplicate_by_corr(top_features, threshold=0.9)

print("\nRecommended features for logistic regression (based on ANOVA F and low redundancy):")
print(", ".join(recommended))

df_plot = data[top_features].copy()
df_plot['house'] = labels['label']

g = sns.PairGrid(df_plot, vars=top_features, hue='house', palette=colors, diag_sharey=False)
g.map_upper(sns.scatterplot, s=18, alpha=0.65, edgecolor='white', linewidth=0.3)
g.map_lower(sns.scatterplot, s=18, alpha=0.65, edgecolor='white', linewidth=0.3)
g.map_diag(sns.histplot, edgecolor='white', linewidth=0.3)
g.add_legend()
g.fig.suptitle(
	'Pair Plot of Top Features by ANOVA F-score (colored by house)',
	fontsize=14,
	fontweight='bold',
	y=1.02,
)
plt.show()
