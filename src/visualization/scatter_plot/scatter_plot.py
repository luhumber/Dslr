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
        description="Plot scatter plots for correlated feature pairs."
    )
    parser.add_argument("-p", "--pair", nargs=2, metavar=("FEATURE1", "FEATURE2"), 
                       help="Plot specific feature pair (ex: --pair 'Astronomy' 'Herbology')")
    parser.add_argument("--top-k", type=int, default=6, help="Number of top pairs to display (default: 6)")
    parser.add_argument("--min-corr", type=float, default=0.0, help="Minimum correlation threshold (default: 0.0)")
    parser.add_argument("--no-show", action="store_true", help="Don't display Matplotlib windows")
    return parser.parse_args()

def calculate_correlations(data: pd.DataFrame, numeric_cols: list[str], min_corr: float = 0.0) -> list[tuple[float, float, str, str]]:
    corr_matrix = data[numeric_cols].corr()
    
    corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            abs_corr = abs(corr_val)
            if abs_corr >= min_corr:
                corr_pairs.append((abs_corr, corr_val, numeric_cols[i], numeric_cols[j]))
    
    corr_pairs.sort(reverse=True)
    return corr_pairs

def plot_correlation_grid(data: pd.DataFrame, labels: pd.DataFrame, houses: np.ndarray, 
                         corr_pairs: list[tuple[float, float, str, str]], top_k: int, show: bool = True) -> None:
    if not corr_pairs:
        print("No correlation pairs to display.")
        return
    
    pairs_to_plot = corr_pairs[:top_k]
    if len(pairs_to_plot) == 0:
        print("No pairs meet the criteria.")
        return
    
    if top_k <= 3:
        nrows, ncols = 1, top_k
    elif top_k <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = int(np.ceil(top_k / 3)), 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 6), facecolor='white')
    fig.suptitle(f'Top {len(pairs_to_plot)} Most Correlated Feature Pairs', fontsize=16, fontweight='bold', y=0.98)
    
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).ravel()
    
    for idx, (abs_corr, corr_val, feat1, feat2) in enumerate(pairs_to_plot):
        ax = axes[idx]
        ax.set_facecolor('#f8f9fa')
        
        for house in houses:
            mask = labels['label'] == house
            ax.scatter(data.loc[mask, feat1], data.loc[mask, feat2],
                      c=HOUSE_COLORS[house], alpha=0.6, s=25, edgecolors='white', 
                      linewidth=0.3, label=house if idx == 0 else "")
        
        set_title_and_labels(ax, f'{feat1} vs {feat2}\nCorrelation: {corr_val:.3f}', 
                            feat1, feat2, title_fontsize=11, label_fontsize=9)
        ax.tick_params(labelsize=8)
        apply_grid(ax)
        
        if abs_corr > 0.8:
            border_color = '#e74c3c'
        elif abs_corr > 0.6:
            border_color = '#f39c12'
        else:
            border_color = '#3498db'
            
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
        
        if idx == 0:
            create_legend(ax, fontsize=9)
    
    for idx in range(len(pairs_to_plot), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if show:
        plt.show()
    else:
        plt.close()

def plot_single_pair(data: pd.DataFrame, labels: pd.DataFrame, houses: np.ndarray, 
                    feat1: str, feat2: str, corr_val: float, show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    for house in houses:
        mask = labels['label'] == house
        ax.scatter(
            data.loc[mask, feat1], data.loc[mask, feat2],
            c=HOUSE_COLORS[house], label=house, alpha=0.7, s=30,
            edgecolors='white', linewidth=0.5
        )
    
    set_title_and_labels(ax, f'Feature Pair: {feat1} vs {feat2} (r={corr_val:.3f})', 
                        feat1, feat2, title_fontsize=14, label_fontsize=12)
    create_legend(ax, fontsize=10)
    apply_grid(ax, alpha=0.2)
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
        print(f"[scatter][ERR] Error loading data: {e}", file=sys.stderr)
        return 2
    
    setup_plot_style()
    numeric_cols = get_numeric_columns(data)
    
    if len(numeric_cols) < 2:
        print("[scatter][ERR] Need at least 2 numeric features for correlation analysis.", file=sys.stderr)
        return 2
    
    if args.pair:
        feat1, feat2 = args.pair
        if feat1 not in numeric_cols or feat2 not in numeric_cols:
            print(f"[scatter][ERR] Unknown features: {feat1}, {feat2}", file=sys.stderr)
            print("Available numeric features:", ", ".join(numeric_cols), file=sys.stderr)
            return 2
        
        corr_val = data[feat1].corr(data[feat2])
        print(f"[scatter] Selected pair: {feat1} vs {feat2} (r={corr_val:.4f})")
        plot_single_pair(data, labels, houses, feat1, feat2, corr_val, not args.no_show)
    else:
        corr_pairs = calculate_correlations(data, numeric_cols, args.min_corr)
        
        if not corr_pairs:
            print(f"[scatter][ERR] No feature pairs found with correlation >= {args.min_corr}", file=sys.stderr)
            return 2
        
        print(f"Top 10 most correlated pairs (min_corr >= {args.min_corr}):")
        for i, (abs_corr, corr_val, feat1, feat2) in enumerate(corr_pairs[:10]):
            print(f"{i+1:2d}. {feat1} vs {feat2}: {corr_val:.3f}")
        
        plot_correlation_grid(data, labels, houses, corr_pairs, args.top_k, not args.no_show)
        
        if corr_pairs:
            print("[scatter] Showing the single most correlated pair...")
            abs_corr, corr_val, feat1, feat2 = corr_pairs[0]
            plot_single_pair(data, labels, houses, feat1, feat2, corr_val, not args.no_show)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
