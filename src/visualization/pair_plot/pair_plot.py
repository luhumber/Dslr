#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from common import (
    load_data, get_numeric_columns, HOUSE_COLORS,
    setup_plot_style
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pair plots for feature analysis and selection."
    )
    parser.add_argument("-f", "--features", nargs="+", help="Specific features to plot (ex: --features 'Astronomy' 'Herbology')")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top features by F-score (default: 5)")
    parser.add_argument("--corr-threshold", type=float, default=0.9, help="Correlation threshold for deduplication (default: 0.9)")
    parser.add_argument("--no-show", action="store_true", help="Don't display Matplotlib windows")
    parser.add_argument("--no-dedup", action="store_true", help="Skip correlation-based deduplication")
    return parser.parse_args()

def anova_fscore(x: np.ndarray, y: np.ndarray, houses: np.ndarray) -> float:
    """Calculate ANOVA F-score for feature discrimination between houses."""
    # Remove invalid values
    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return 0.0
    
    # Group data by house
    groups = []
    for house in houses:
        house_mask = y == house
        house_data = x[house_mask]
        if len(house_data) > 0:
            groups.append(house_data)
    
    if len(groups) <= 1:
        return 0.0
    
    sizes = [len(g) for g in groups]
    
    if min(sizes) == 0 or np.std(x) == 0:
        return 0.0
    
    overall_mean = np.mean(x)
    
    # Between-group sum of squares
    ss_between = sum(n_k * (np.mean(g) - overall_mean) ** 2 for g, n_k in zip(groups, sizes))
    
    # Within-group sum of squares
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    
    k = len(groups)  # number of groups
    n = sum(sizes)   # total observations
    
    if k <= 1 or n <= k or ss_within == 0:
        return 0.0
    
    # Calculate F-statistic
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    
    return float(ms_between / ms_within) if ms_within > 0 else 0.0

def calculate_feature_scores(data: pd.DataFrame, labels: pd.DataFrame, houses: np.ndarray, numeric_cols: list[str]) -> list[tuple[str, float]]:
    """Calculate ANOVA F-scores for all numeric features."""
    fscores = {}
    y_labels = labels['label'].to_numpy()
    
    for col in numeric_cols:
        x_values = data[col].to_numpy()
        fscores[col] = anova_fscore(x_values, y_labels, houses)
    
    # Sort by F-score descending
    ranked = sorted(fscores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked

def deduplicate_by_corr(data: pd.DataFrame, features: list[str], threshold: float = 0.9) -> list[str]:
    """Remove highly correlated features to reduce redundancy."""
    if len(features) <= 1:
        return features
    
    kept = []
    corr = data[features].corr().abs()
    
    for f in features:
        # Check if this feature is highly correlated with any kept feature
        if all(corr.loc[f, k] < threshold for k in kept if k in corr.columns):
            kept.append(f)
    
    return kept

def create_pair_plot(data: pd.DataFrame, labels: pd.DataFrame, features: list[str], show: bool = True) -> None:
    """Create seaborn pair plot for given features."""
    if len(features) == 0:
        print("No features to plot.")
        return
    
    if len(features) > 6:
        print(f"Warning: Plotting {len(features)} features may be slow. Consider using --top-k to limit.")
    
    # Prepare data for seaborn
    df_plot = data[features].copy()
    df_plot['house'] = labels['label']
    
    # Create pair plot
    g = sns.PairGrid(df_plot, vars=features, hue='house', palette=HOUSE_COLORS, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=18, alpha=0.65, edgecolor='white', linewidth=0.3)
    g.map_lower(sns.scatterplot, s=18, alpha=0.65, edgecolor='white', linewidth=0.3)
    g.map_diag(sns.histplot, edgecolor='white', linewidth=0.3)
    g.add_legend()
    
    title = f'Pair Plot of Features (n={len(features)})'
    g.figure.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    if show:
        plt.show()
    else:
        plt.close()

def main() -> int:
    args = parse_args()
    
    try:
        data, labels, houses = load_data()
    except Exception as e:
        print(f"[pair_plot][ERR] Error loading data: {e}", file=sys.stderr)
        return 2
    
    setup_plot_style()
    numeric_cols = get_numeric_columns(data)
    
    if len(numeric_cols) < 2:
        print("[pair_plot][ERR] Need at least 2 numeric features for pair plot.", file=sys.stderr)
        return 2
    
    if args.features:
        # User specified features
        unknown_features = [f for f in args.features if f not in numeric_cols]
        if unknown_features:
            print(f"[pair_plot][ERR] Unknown features: {', '.join(unknown_features)}", file=sys.stderr)
            print("Available numeric features:", ", ".join(numeric_cols), file=sys.stderr)
            return 2
        
        selected_features = args.features
        print(f"[pair_plot] Using user-specified features: {', '.join(selected_features)}")
    else:
        # Automatic feature selection based on ANOVA F-score
        ranked = calculate_feature_scores(data, labels, houses, numeric_cols)
        
        if not ranked or all(score == 0 for _, score in ranked):
            print("[pair_plot][ERR] No discriminative features found.", file=sys.stderr)
            return 2
        
        print("Top 10 features by ANOVA F-score (descending):")
        for i, (feat, f) in enumerate(ranked[:10], start=1):
            print(f"{i:2d}. {feat}: F = {f:.3f}")
        
        # Select top features
        top_features = [feat for feat, _ in ranked[:args.top_k]]
        
        if not args.no_dedup:
            # Remove highly correlated features
            selected_features = deduplicate_by_corr(data, top_features, args.corr_threshold)
            if len(selected_features) < len(top_features):
                removed = set(top_features) - set(selected_features)
                print(f"\nRemoved highly correlated features (r > {args.corr_threshold}): {', '.join(removed)}")
        else:
            selected_features = top_features
        
        print(f"\nRecommended features for analysis (n={len(selected_features)}):")
        print(", ".join(selected_features))
    
    # Create pair plot
    create_pair_plot(data, labels, selected_features, not args.no_show)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
