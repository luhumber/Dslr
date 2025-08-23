import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from common import (
    load_data, get_numeric_columns, HOUSE_COLORS,
    setup_plot_style, create_legend, apply_grid, set_title_and_labels
)

data, labels, houses = load_data()
numeric_cols = get_numeric_columns(data)
setup_plot_style()

corr_matrix = data[numeric_cols].corr()

upper_triangle = np.triu(corr_matrix, k=1)
max_corr_idx = np.unravel_index(np.argmax(np.abs(upper_triangle)), upper_triangle.shape)
feature1, feature2 = numeric_cols[max_corr_idx[0]], numeric_cols[max_corr_idx[1]]

corr_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        corr_val = corr_matrix.iloc[i, j]
        corr_pairs.append((abs(corr_val), corr_val, numeric_cols[i], numeric_cols[j]))

corr_pairs.sort(reverse=True)

print("Top 10 most correlated pairs:")
for i, (abs_corr, corr_val, feat1, feat2) in enumerate(corr_pairs[:10]):
    print(f"{i+1}. {feat1} vs {feat2}: {corr_val:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
fig.suptitle('Top 6 Most Correlated Feature Pairs', fontsize=16, fontweight='bold', y=0.98)
axes = axes.flatten()

for idx, (abs_corr, corr_val, feat1, feat2) in enumerate(corr_pairs[:6]):
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

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show(block=True)

print("[scatter] Showing the single most correlated pair first…")
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
for house in houses:
    mask = labels['label'] == house
    ax.scatter(
        data.loc[mask, feature1], data.loc[mask, feature2],
        c=HOUSE_COLORS[house], label=house, alpha=0.7, s=30,
        edgecolors='white', linewidth=0.5
    )

set_title_and_labels(ax, f'Most Correlated Features: {feature1} vs {feature2}', 
                    feature1, feature2, title_fontsize=14, label_fontsize=12)
create_legend(ax, fontsize=10)
apply_grid(ax, alpha=0.2)
plt.tight_layout()
plt.show(block=True)
