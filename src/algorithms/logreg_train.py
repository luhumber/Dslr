import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    selected_features = [
        'Arithmancy',
        'Defense Against the Dark Arts',
        'Herbology',
        'Potions',
        'Transfiguration'
    ]
    
    features = df[selected_features]
    target = df['Hogwarts House']
    
    features = features.fillna(features.median())
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    target_binary = {}
    
    for house in houses:
        target_binary[house] = (target == house).astype(int)
    
    return features_scaled, target_binary, scaler

def sigmoid(z):
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

def train_logistic_regression(features, target, learning_rate=0.01, max_iterations=1000):
    features_with_bias = np.column_stack([np.ones(features.shape[0]), features])

    num_weights = features_with_bias.shape[1]
    weights = np.random.normal(0, 0.01, num_weights)

    for i in range(max_iterations):
        z = features_with_bias.dot(weights)
        target_predicted = sigmoid(z)

        cost = cost_function(target, target_predicted)

        gradient = compute_gradient(features_with_bias, target, target_predicted)

        weights = weights - learning_rate * gradient

        if i % 200 == 0:
            print(f"Iteration {i}, Cost: {cost:.4f}")

    return weights

features_scaled, target_binary, scaler = load_and_preprocess_data('datasets/dataset_train.csv')

print("Training logistic regression for each house...")

houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
weights = {}

for house in houses:
    print(f"Training for {house}...")

    weights[house] = train_logistic_regression(
        features_scaled,
        target_binary[house],
        learning_rate=0.01,
        max_iterations=1000
    )

    print(f"Training complete for {house}.")

print("Training complete for all houses.")

if not os.path.exists('output'):
    os.makedirs('output')

data_to_save = {
    "weights": weights,
    "scaler": scaler
}

with open('output/model.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Model saved to 'output/model.pkl'")