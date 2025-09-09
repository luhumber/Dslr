import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse
import sys

def load_and_preprocess_data(file_path, validation_split=0.2, full_dataset=False):
    df = pd.read_csv(file_path)
    
    selected_features = [
        'Charms',
        'Defense Against the Dark Arts',
        'Ancient Runes',
        'Herbology',
        'Divination',
    ]
    
    features = df[selected_features]
    target = df['Hogwarts House']
    
    features = features.fillna(features.median())
    
    if full_dataset:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
        target_binary = {}
        
        for house in houses:
            target_binary[house] = (target == house).astype(int)
        
        return features_scaled, target_binary, scaler
    
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * (1 - validation_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    features_train = features.iloc[train_indices]
    features_val = features.iloc[val_indices]
    target_train = target.iloc[train_indices]
    target_val = target.iloc[val_indices]
    
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_val_scaled = scaler.transform(features_val)
    
    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    target_train_binary = {}
    target_val_binary = {}
    
    for house in houses:
        target_train_binary[house] = (target_train == house).astype(int)
        target_val_binary[house] = (target_val == house).astype(int)
    
    return (features_train_scaled, target_train_binary, 
            features_val_scaled, target_val_binary, scaler)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def cost_function(real_target, predicted_target):
    epsilon = 1e-15
    predicted_target = np.clip(predicted_target, epsilon, 1 - epsilon)
    cost = -np.mean(real_target * np.log(predicted_target) + (1 - real_target) * np.log(1 - predicted_target))
    return cost

def compute_gradient(features, real_target, predicted_target):
    m = features.shape[0]
    gradient = (1/m) * features.T.dot(predicted_target - real_target)
    return gradient

def train_logistic_regression(features, target, learning_rate=0.001, max_iterations=5000):
    features_with_bias = np.column_stack([np.ones(features.shape[0]), features])

    num_weights = features_with_bias.shape[1]
    np.random.seed(42)
    weights = np.random.normal(0, 0.01, num_weights)

    prev_cost = float('inf')
    tolerance = 1e-6

    for i in range(max_iterations):
        z = features_with_bias.dot(weights)
        target_predicted = sigmoid(z)

        cost = cost_function(target, target_predicted)

        gradient = compute_gradient(features_with_bias, target, target_predicted)

        weights = weights - learning_rate * gradient

        if i % 500 == 0:
            print(f"Iteration {i}, Cost: {cost:.6f}")

        if abs(prev_cost - cost) < tolerance:
            print(f"Converged at iteration {i}, Cost: {cost:.6f}")
            break
        prev_cost = cost

    return weights

def parse_args():
    parser = argparse.ArgumentParser(description='Train logistic regression model for Hogwarts house prediction')
    parser.add_argument('dataset_path', nargs='?', default='datasets/dataset_train.csv',
                       help='Path to training dataset CSV file (default: datasets/dataset_train.csv)')
    parser.add_argument('--output', '-o', default='output/model.pkl',
                       help='Output path for trained model (default: output/model.pkl)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file '{args.dataset_path}' not found", file=sys.stderr)
        return 1
    
    try:
        result = load_and_preprocess_data(args.dataset_path, validation_split=0.2, full_dataset=False)
        if len(result) != 5:
            print("Error: Unexpected return format from load_and_preprocess_data", file=sys.stderr)
            return 1
        features_train, target_train_binary, features_val, target_val_binary, scaler = result
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    print(f"Training set shape: {features_train.shape}")
    print(f"Validation set shape: {features_val.shape}")
    print(f"Training features means: {features_train.mean(axis=0)}")
    print(f"Training features stds: {features_train.std(axis=0)}")

    print("Training logistic regression for each house...")

    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    weights = {}

    for house in houses:
        print(f"Training for {house}...")

        weights[house] = train_logistic_regression(
            features_train,
            target_train_binary[house]
        )

        print(f"Training complete for {house}.")

    print("Training complete for all houses.")

    print("\n=== Testing on validation set ===")
    from collections import Counter

    val_predictions = []
    for student_features in features_val:
        features_with_bias = np.concatenate(([1], student_features))
        scores = {}
        
        for house in houses:
            score = features_with_bias.dot(weights[house])
            probability = sigmoid(score)
            scores[house] = probability
        
        best_house = max(scores.items(), key=lambda x: x[1])[0]
        val_predictions.append(best_house)

    val_target_true = []
    for i in range(len(features_val)):
        for house in houses:
            if target_val_binary[house].iloc[i] == 1:
                val_target_true.append(house)
                break

    correct = sum(1 for pred, true in zip(val_predictions, val_target_true) if pred == true)
    val_accuracy = correct / len(val_predictions) * 100

    print(f"Validation Accuracy: {correct}/{len(val_predictions)} = {val_accuracy:.2f}%")

    pred_counts = Counter(val_predictions)
    print("Validation prediction distribution:")
    for house, count in pred_counts.items():
        print(f"  {house}: {count} ({count/len(val_predictions)*100:.1f}%)")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_to_save = {
        "weights": weights,
        "scaler": scaler
    }

    with open(args.output, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Model saved to '{args.output}'")
    return 0

if __name__ == "__main__":
    sys.exit(main())