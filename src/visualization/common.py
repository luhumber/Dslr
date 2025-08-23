import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DATA_PATH = 'data/train/dataset_clean.csv'
LABELS_PATH = 'data/train/labels.csv'

HOUSE_COLORS = {
    'Gryffindor': '#E74C3C',
    'Hufflepuff': '#F39C12',
    'Ravenclaw': '#3498DB',
    'Slytherin': '#27AE60'
}

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data = pd.read_csv(DATA_PATH)
    labels = pd.read_csv(LABELS_PATH)
    houses = labels['label'].unique()
    return data, labels, houses

def get_numeric_columns(data: pd.DataFrame) -> list[str]:
    return data.select_dtypes(include=[np.number]).columns.tolist()

def setup_plot_style(figure_params: Optional[Dict] = None) -> None:
    if figure_params is None:
        figure_params = {'facecolor': 'white'}
    
    plt.rcParams.update({
        'figure.facecolor': figure_params.get('facecolor', 'white'),
        'axes.facecolor': '#f8f9fa',
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def create_legend(ax, fontsize: int = 9) -> None:
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=fontsize)

def apply_grid(ax, alpha: float = 0.3) -> None:
    ax.grid(True, alpha=alpha, linestyle='--')

def set_title_and_labels(ax, title: str, xlabel: str, ylabel: str, 
                        title_fontsize: int = 11, label_fontsize: int = 9) -> None:
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold')