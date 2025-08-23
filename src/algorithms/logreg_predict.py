import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def load_model(model_file):
    with open(model_file, 'rb') as f:
        data = pickle.load(f)
    return data['weights'], data['scaler']

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_house(student_features, weights):
    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    scores = {}
    
    features_with_bias = np.concatenate(([1], student_features))
    
    for house in houses:
        score = features_with_bias.dot(weights[house])
        probability = sigmoid(score)
        scores[house] = probability
    
    best_house = None
    best_score = 0
    
    for house, probability in scores.items():
        if probability > best_score:
            best_score = probability
            best_house = house
    
    return best_house

def load_and_preprocess_test_data(test_file, scaler):
    df = pd.read_csv(test_file)
    
    selected_features = [
        'Arithmancy',
        'Defense Against the Dark Arts',
        'Herbology',
        'Potions',
        'Transfiguration'
    ]
    
    features = df[selected_features]
    features = features.fillna(features.median())
    features_scaled = scaler.transform(features)
    
    return features_scaled

def make_predictions(test_file, model_file):
    weights, scaler = load_model(model_file)
    
    features_scaled = load_and_preprocess_test_data(test_file, scaler)
    
    predictions = []
    
    for i, student_features in enumerate(features_scaled):
        predicted_house = predict_house(student_features, weights)
        predictions.append(predicted_house)
        
        if i < 5:
            print(f"Student {i+1}: {predicted_house}")
    
    print(f"\nPredictions completed for {len(predictions)} students!")
    
    results_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })
    
    results_df.to_csv('output/houses.csv', index=False)
    print("Predictions saved to 'houses.csv'")
    
    return predictions

if __name__ == "__main__":
    predictions = make_predictions('datasets/dataset_test.csv', 'output/model.pkl')