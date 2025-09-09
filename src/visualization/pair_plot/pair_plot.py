from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
    parser.add_argument("--top-k", type=int, default=100, help="Number of top features by F-score (default: 100)")
    parser.add_argument("--no-show", action="store_true", help="Don't display Matplotlib windows")
    return parser.parse_args()

def anova_fscore(x: np.ndarray, y: np.ndarray, houses: np.ndarray) -> float:
    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return 0.0
    
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
    
    ss_between = sum(n_k * (np.mean(g) - overall_mean) ** 2 for g, n_k in zip(groups, sizes))
    
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    
    k = len(groups)
    n = sum(sizes) 
    
    if k <= 1 or n <= k or ss_within == 0:
        return 0.0
    
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    
    return float(ms_between / ms_within) if ms_within > 0 else 0.0

def calculate_feature_scores(data: pd.DataFrame, labels: pd.DataFrame, houses: np.ndarray, numeric_cols: list[str]) -> list[tuple[str, float]]:
    fscores = {}
    y_labels = labels['label'].to_numpy()
    
    for col in numeric_cols:
        x_values = data[col].to_numpy()
        fscores[col] = anova_fscore(x_values, y_labels, houses)
    
    ranked = sorted(fscores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked


def create_pair_plot(data: pd.DataFrame, labels: pd.DataFrame, features: list[str], show: bool = True) -> None:
    if len(features) == 0:
        print("No features to plot.")
        return
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return
    
    df_plot = data[features].copy()
    df_plot['house'] = labels['label']
    
    house_color_map = HOUSE_COLORS
    
    n_features = len(features)
    
    spacing = max(0.01, min(0.05, 0.3 / n_features))
    
    fig = make_subplots(
        rows=n_features, cols=n_features,
        subplot_titles=features if n_features <= 8 else None,
        vertical_spacing=spacing, horizontal_spacing=spacing
    )
    
    houses = df_plot['house'].unique()
    
    def add_scatter_plot(fig, x_data, y_data, house, row, col, show_legend=False):
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name=house,
                legendgroup=house,
                showlegend=show_legend,
                marker=dict(
                    color=house_color_map[house],
                    size=3,
                    opacity=0.6
                )
            ),
            row=row, col=col
        )
    
    legend_shown = set()
    
    house_data_dict = {house: df_plot[df_plot['house'] == house] for house in houses}
    
    for i, feat_x in enumerate(features):
        for j, feat_y in enumerate(features):
            row, col = i + 1, j + 1
            
            if i == j:
                for house in houses:
                    house_data = house_data_dict[house]
                    show_legend = house not in legend_shown
                    if show_legend:
                        legend_shown.add(house)
                    
                    fig.add_trace(
                        go.Histogram(
                            x=house_data[feat_x],
                            name=house,
                            legendgroup=house,
                            showlegend=show_legend,
                            marker_color=house_color_map[house],
                            opacity=0.7,
                            nbinsx=15
                        ),
                        row=row, col=col
                    )
            else:
                for house in houses:
                    house_data = house_data_dict[house]
                    show_legend = house not in legend_shown
                    if show_legend:
                        legend_shown.add(house)
                    add_scatter_plot(
                        fig, house_data[feat_x], house_data[feat_y], 
                        house, row, col, show_legend
                    )
    
    plot_size = 200
    font_size = max(8, min(12, 80 / n_features))
    
    fig.update_layout(
        title=f'Interactive Pair Plot (n={len(features)} features)',
        width=plot_size * n_features,
        height=plot_size * n_features,
        font={'size': font_size},
        title_x=0.5,
        barmode='overlay',
        showlegend=True
    )
    
    for i in range(n_features):
        fig.update_xaxes(title_text=features[i], row=n_features, col=i+1)
        fig.update_yaxes(title_text=features[i], row=i+1, col=1)
    
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, ticks="")
    
    if show:
        fig.show()
    else:
        fig.write_html("pair_plot.html")
        print("Interactive plot saved as pair_plot.html")

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
        unknown_features = [f for f in args.features if f not in numeric_cols]
        if unknown_features:
            print(f"[pair_plot][ERR] Unknown features: {', '.join(unknown_features)}", file=sys.stderr)
            print("Available numeric features:", ", ".join(numeric_cols), file=sys.stderr)
            return 2
        
        selected_features = args.features
        print(f"[pair_plot] Using user-specified features: {', '.join(selected_features)}")
    else:
        ranked = calculate_feature_scores(data, labels, houses, numeric_cols)
        
        if not ranked or all(score == 0 for _, score in ranked):
            print("[pair_plot][ERR] No discriminative features found.", file=sys.stderr)
            return 2
        
        print("Top 10 features by ANOVA F-score (descending):")
        for i, (feat, f) in enumerate(ranked[:10], start=1):
            print(f"{i:2d}. {feat}: F = {f:.3f}")
        
        top_features = [feat for feat, _ in ranked[:args.top_k]]
        
        selected_features = top_features
    
    create_pair_plot(data, labels, selected_features, not args.no_show)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
