from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from common import (
    load_data, get_numeric_columns, HOUSE_COLORS,
    setup_plot_style, create_legend, apply_grid, set_title_and_labels
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histograms by house for a given course."
    )
    parser.add_argument("-f", "--feature", help="Exact column name (ex: Astronomy)")
    parser.add_argument("--bins", type=int, default=24, help="Number of bins (default: 24)")
    parser.add_argument("--no-show", action="store_true", help="Don't display Matplotlib windows")
    parser.add_argument("--top-k", type=int, default=6, help="Number of features to display (default: 6)")
    return parser.parse_args()

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

def calculate_homogeneity(data: pd.DataFrame, labels: pd.DataFrame, houses: np.ndarray, bins: int) -> dict[str, float]:
	numeric_courses = get_numeric_columns(data)
	homogeneity = {}

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

	return homogeneity

def plot_top_features(data: pd.DataFrame, labels: pd.DataFrame, houses: np.ndarray, 
                     to_plot: list[tuple[str, float]], bins: int, show: bool = True) -> None:
	if not to_plot:
		print("No numeric courses to display.")
		return

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
				color=HOUSE_COLORS.get(house, None),
				edgecolor='white',
				linewidth=0.5,
			)

		set_title_and_labels(ax, f"{course}\nJS_mean={score:.3f}", 'Score', "# observations")
		apply_grid(ax)
		create_legend(ax, fontsize=8)

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
	if show:
		plt.show()
	else:
		plt.close()

def plot_best_feature(data: pd.DataFrame, labels: pd.DataFrame, houses: np.ndarray, 
                     best_course: str, best_score: float, bins: int, show: bool = True) -> None:
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
			color=HOUSE_COLORS.get(house, None),
			edgecolor='white',
			linewidth=0.6,
		)

	plt.title(
		f"Most homogeneous course between the four houses: {best_course} (JS_mean={best_score:.3f})",
		fontsize=13, fontweight='bold', pad=12
	)
	plt.xlabel('Score', fontsize=10, fontweight='bold')
	plt.ylabel('# observations', fontsize=10, fontweight='bold')
	apply_grid(plt.gca())
	plt.legend(frameon=True, fancybox=True, fontsize=9)
	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.close()

def main() -> int:
	args = parse_args()
	
	try:
		data, labels, houses = load_data()
	except Exception as e:
		print(f"[histogram][ERR] Error loading data: {e}", file=sys.stderr)
		return 2
	
	setup_plot_style()
	
	homogeneity = calculate_homogeneity(data, labels, houses, args.bins)
	
	if not homogeneity:
		print("[histogram][ERR] No homogeneous numeric features found.", file=sys.stderr)
		return 2
	
	ranked = sorted(homogeneity.items(), key=lambda kv: kv[1])
	
	if args.feature:
		if args.feature not in homogeneity:
			print(f"[histogram][ERR] Unknown or non-numeric feature: '{args.feature}'.", file=sys.stderr)
			print("Available numeric features:", ", ".join(homogeneity.keys()), file=sys.stderr)
			return 2
		
		score = homogeneity[args.feature]
		print(f"[histogram] Selected feature: {args.feature} (JS_mean={score:.4f})")
		plot_best_feature(data, labels, houses, args.feature, score, args.bins, not args.no_show)
	else:
		print("Top 10 most homogeneous courses (Jensen-Shannon mean increasing):")
		for i, (course, score) in enumerate(ranked[:10], start=1):
			print(f"{i:2d}. {course}: JS_mean = {score:.4f}")
		
		to_plot = ranked[:args.top_k]
		plot_top_features(data, labels, houses, to_plot, args.bins, not args.no_show)
		
		if ranked:
			best_course, best_score = ranked[0]
			plot_best_feature(data, labels, houses, best_course, best_score, args.bins, not args.no_show)
	
	return 0

if __name__ == "__main__":
	raise SystemExit(main())
