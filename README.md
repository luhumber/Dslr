# DSLR - Data Science Logistic Regression

A complete machine learning pipeline that implements the Hogwarts Sorting Hat using logistic regression. This project predicts which Hogwarts house a student belongs to based on their magical course grades.

## 🎯 Project Overview

**Goal**: Create a logistic regression model that achieves 98%+ accuracy on the test set to replicate the Sorting Hat's decision-making process.

**Approach**: One-vs-All multiclass classification with 4 binary logistic regression models (one per house).

## 🚀 Quick Start

### Setup
```bash
# Install dependencies
make deps

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train & Predict
```bash
# Train model on all 1600 samples
make train-model

# Generate predictions for test set
make predict-model

# Results saved to: output/houses.csv
```

## 📊 Data Analysis

### Descriptive Statistics
```bash
# Analyze training dataset
make describe-train

# Analyze test dataset  
make describe-test

# Custom dataset
make describe datasets/my_data.csv
```

The `describe` script manually computes statistics (Count, Mean, Std, Min, 25%, 50%, 75%, Max) for all numeric columns without using pandas/numpy built-in functions.

### Data Preprocessing
```bash
# Prepare cleaned datasets
make prepare-train
make prepare-test

# Inspect processed data
make inspect-train
make inspect-test
```

## 📈 Visualizations

### All Visualizations
```bash
make visualizer          # Run all visualizations
```

### Individual Visualizations
```bash
# Histogram analysis - find most homogeneous course
make visualizer-hist

# Scatter plots - explore feature correlations  
make visualizer-scatter

# Pair plots - comprehensive feature relationships
make visualizer-pair
```

## 🤖 Machine Learning Pipeline

### Training
```bash
# Default training (uses datasets/dataset_train.csv)
make train-model

# Custom dataset training
make train-custom dataset=path/to/data.csv

# Custom output path
make train-custom dataset=path/to/data.csv output=path/to/model.pkl
```

**Features Used**:
- Charms (F-score: 3663)
- Defense Against the Dark Arts (F-score: 3594)  
- Ancient Runes (F-score: 2731)
- Herbology (F-score: 2338)
- Divination (F-score: 2374)

**Algorithm Details**:
- Standardization: Z-score normalization
- Optimization: Gradient descent (learning_rate=0.001)
- Convergence: Early stopping when cost improvement < 1e-6
- Maximum iterations: 5000

### Prediction
```bash
# Predict on test set (default)
make predict-model

# Direct script usage
python src/algorithms/logreg_predict.py
```

**Output**: `output/houses.csv` with format:
```csv
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
2,Gryffindor
```

## 🔧 Advanced Usage

### Custom Feature Selection
Edit `selected_features` in `src/algorithms/logreg_train.py`:

```python
selected_features = [
    'Charms',
    'Defense Against the Dark Arts', 
    'Astronomy',  # Try different features
    'Ancient Runes',
    'Flying',
]
```

### Visualization Options
```bash
# Pair plot with specific features
python src/visualization/pair_plot/pair_plot.py --features "Charms" "Defense Against the Dark Arts"

# Scatter plot with correlation threshold
python src/visualization/scatter_plot/scatter_plot.py --min-corr 0.5

# Histogram for specific course
python src/visualization/histogram/histogram.py --feature "Astronomy"
```

## 📁 Project Structure

```
dslr/
├── datasets/              # Raw CSV data
├── data/                 # Processed datasets and artifacts
├── output/               # Models and predictions
├── src/
│   ├── algorithms/       # ML training & prediction
│   ├── parsing/          # Data preprocessing & analysis
│   └── visualization/    # Data visualization tools
├── Makefile             # Task automation
└── requirements.txt     # Python dependencies
```

## 🧪 Mathematical Foundation

### Logistic Regression
For each house h, we learn weights θ such that:

```
P(house = h | features) = σ(θ₀ + θ₁x₁ + ... + θₙxₙ)
```

Where σ is the sigmoid function: `σ(z) = 1/(1 + e^(-z))`

### Cost Function
Binary cross-entropy loss for each house classifier:

```
J(θ) = -(1/m) Σ[y*log(hθ(x)) + (1-y)*log(1-hθ(x))]
```

### Feature Selection
Features ranked by ANOVA F-score measuring discriminative power between houses:

```
F = (Between-group variance) / (Within-group variance)
```

## 🎯 Performance

- **Validation Accuracy**: 97.19% (320 samples)
- **Expected Test Accuracy**: >98% (with full 1600 training samples)
- **Convergence**: Typically ~4000 iterations per house

## 🛠️ Utilities

```bash
# Clean generated files
make clean

# Help with Makefile commands
make help
```

## 📋 Requirements

- Python 3.8+
- Dependencies: numpy, pandas, matplotlib, seaborn, plotly, scikit-learn

## 🏆 42 School Project

This project replicates the Sorting Hat's decision-making process using machine learning techniques, demonstrating proficiency in:
- Data preprocessing and analysis
- Manual implementation of statistical functions  
- Logistic regression from scratch
- Data visualization and feature engineering
- Model evaluation and hyperparameter tuning

**Target**: Achieve 98%+ accuracy to match the Sorting Hat's performance!