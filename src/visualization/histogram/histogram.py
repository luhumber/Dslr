import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

numeric_courses = data.select_dtypes(include=[np.number]).columns.tolist()

def hist_prob(values: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
	values = values[np.isfinite(values)]
	if values.size == 0:
		return np.array([1.0]), np.array([0.0, 1.0])
	counts, bin_edges = np.histogram(values, bins=bins)
	total = counts.sum()
	if total == 0:
		p = np.ones_like(counts, dtype=float) / len(counts)
	else:
		p = counts.astype(float) / total
	return p, bin_edges

def hist_prob_fixed_bins(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
	counts, _ = np.histogram(values[np.isfinite(values)], bins=bin_edges)
	total = counts.sum()
	if total == 0:
		return np.ones_like(counts, dtype=float) / len(counts)
	return counts.astype(float) / total

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
	p = np.clip(p, eps, 1.0)
	q = np.clip(q, eps, 1.0)
	p = p / p.sum()
	q = q / q.sum()
	m = 0.5 * (p + q)
	kl_pm = np.sum(p * np.log(p / m))
	kl_qm = np.sum(q * np.log(q / m))
	return 0.5 * (kl_pm + kl_qm)

def average_pairwise_js(distributions: list[np.ndarray]) -> float:
	if len(distributions) < 2:
		return 0.0
	js_values = []
	for i in range(len(distributions)):
		for j in range(i + 1, len(distributions)):
			js_values.append(js_divergence(distributions[i], distributions[j]))
	return float(np.mean(js_values)) if js_values else 0.0

homogeneity = {}
bins = 24

for course in numeric_courses:
	course_values = data[course].to_numpy()
	course_values = course_values[np.isfinite(course_values)]

	if course_values.size == 0 or np.nanstd(course_values) == 0:
		continue

	vmin, vmax = np.nanmin(course_values), np.nanmax(course_values)
	if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
		continue
	bin_edges = np.linspace(vmin, vmax, bins + 1)

	dists = []
	for house in houses:
		mask = labels['label'] == house
		vals = data.loc[mask, course].to_numpy()
		p = hist_prob_fixed_bins(vals, bin_edges)
		dists.append(p)

	homogeneity[course] = average_pairwise_js(dists)

ranked = sorted(homogeneity.items(), key=lambda kv: kv[1])

print("Top 10 most homogeneous courses (Jensen–Shannon mean increasing):")
for i, (course, score) in enumerate(ranked[:10], start=1):
	print(f"{i:2d}. {course}: JS_mean = {score:.4f}")

top_k = 6
to_plot = ranked[:top_k]

if to_plot:
	ncols = 3
	nrows = int(np.ceil(len(to_plot) / ncols))
	fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 4.5), facecolor='white')
	axes = np.atleast_1d(axes).ravel()

	for ax, (course, score) in zip(axes, to_plot):
		course_values = data[course].to_numpy()
		course_values = course_values[np.isfinite(course_values)]
		vmin, vmax = np.nanmin(course_values), np.nanmax(course_values)
		bin_edges = np.linspace(vmin, vmax, bins + 1)

		for house in houses:
			mask = labels['label'] == house
			vals = data.loc[mask, course].to_numpy()
			ax.hist(
				vals[np.isfinite(vals)],
				bins=bin_edges,
				alpha=0.55,
				label=house,
				color=colors.get(house, None),
				edgecolor='white',
				linewidth=0.5,
			)

		ax.set_title(f"{course}\nJS_mean={score:.3f}", fontsize=11, fontweight='bold', pad=10)
		ax.set_xlabel('Score', fontsize=9, fontweight='bold')
		ax.set_ylabel("# observations", fontsize=9, fontweight='bold')
		ax.grid(True, alpha=0.3, linestyle='--')
		ax.legend(fontsize=8, frameon=True, fancybox=True, shadow=False)

	for ax in axes[len(to_plot):]:
		ax.axis('off')

	fig.suptitle(
		'Top homogeneous courses (overlaid histograms by house)',
		fontsize=14,
		fontweight='bold',
		y=0.98,
	)
	plt.tight_layout()
	plt.subplots_adjust(top=0.9)
	plt.show()
else:
	print("No numeric courses to display.")

if ranked:
	best_course, best_score = ranked[0]

	course_values = data[best_course].to_numpy()
	course_values = course_values[np.isfinite(course_values)]
	vmin, vmax = np.nanmin(course_values), np.nanmax(course_values)
	bin_edges = np.linspace(vmin, vmax, bins + 1)

	plt.figure(figsize=(8, 5), facecolor='white')
	for house in houses:
		mask = labels['label'] == house
		vals = data.loc[mask, best_course].to_numpy()
		plt.hist(
			vals[np.isfinite(vals)],
			bins=bin_edges,
			alpha=0.6,
			label=house,
			color=colors.get(house, None),
			edgecolor='white',
			linewidth=0.6,
		)

	plt.title(
		f"Most homogeneous course between the four houses: {best_course} (JS_mean={best_score:.3f})",
		fontsize=13,
		fontweight='bold',
		pad=12,
	)
	plt.xlabel('Score', fontsize=10, fontweight='bold')
	plt.ylabel('# observations', fontsize=10, fontweight='bold')
	plt.grid(True, alpha=0.3, linestyle='--')
	plt.legend(frameon=True, fancybox=True, fontsize=9)
	plt.tight_layout()
	plt.show()
