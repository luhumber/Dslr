import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

data = pd.read_csv('data/train/dataset_clean.csv')
labels = pd.read_csv('data/train/labels.csv')

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

corr_matrix = data[numeric_cols].corr()

upper_triangle = np.triu(corr_matrix, k=1)
max_corr_idx = np.unravel_index(np.argmax(np.abs(upper_triangle)), upper_triangle.shape)
feature1, feature2 = numeric_cols[max_corr_idx[0]], numeric_cols[max_corr_idx[1]]

colors = {
    'Gryffindor': '#E74C3C',
    'Hufflepuff': '#F39C12',
    'Ravenclaw': '#3498DB',
    'Slytherin': '#27AE60'
}

plt.figure(figsize=(12, 8), facecolor='white')
for house in labels['label'].unique():
    mask = labels['label'] == house
    plt.scatter(data.loc[mask, feature1], data.loc[mask, feature2],
                c=colors[house], label=house, alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

plt.xlabel(feature1, fontsize=12, fontweight='bold')
plt.ylabel(feature2, fontsize=12, fontweight='bold')
plt.title(f'Most Correlated Features: {feature1} vs {feature2}', fontsize=14, fontweight='bold', pad=20)
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
plt.grid(True, alpha=0.2, linestyle='--')
plt.tight_layout()
plt.show()

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
    
    for house in labels['label'].unique():
        mask = labels['label'] == house
        ax.scatter(data.loc[mask, feat1], data.loc[mask, feat2],
                  c=colors[house], alpha=0.6, s=25, edgecolors='white', 
                  linewidth=0.3, label=house if idx == 0 else "")
    
    ax.set_title(f'{feat1} vs {feat2}\nCorrelation: {corr_val:.3f}', 
                fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel(feat1, fontsize=9, fontweight='bold')
    ax.set_ylabel(feat2, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    
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
        ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
