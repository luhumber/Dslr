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
    
    return max(scores.items(), key=lambda x: x[1])[0]

def load_and_preprocess_test_data(test_file, scaler):
    df = pd.read_csv(test_file)
    
    selected_features = [
        'Charms',
        'Defense Against the Dark Arts',
        'Ancient Runes',
        'Herbology',
        'Divination'
    ]
    
    features = df[selected_features]
    features = features.fillna(features.median())
    features_scaled = scaler.transform(features)
    
    return features_scaled

def make_predictions(test_file, model_file):
    weights, scaler = load_model(model_file)
    features_scaled = load_and_preprocess_test_data(test_file, scaler)
    
    predictions = []
    for student_features in features_scaled:
        predicted_house = predict_house(student_features, weights)
        predictions.append(predicted_house)
    
    
    results_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })
    
    results_df.to_csv('output/houses.csv', index=False)
    print(f"Predictions saved to 'output/houses.csv'")
    
    return predictions

def predict_all_students():
    weights, scaler = load_model('output/model.pkl')
    features_scaled = load_and_preprocess_test_data('datasets/dataset_test.csv', scaler)
    
    predictions = []
    for student_features in features_scaled:
        predicted_house = predict_house(student_features, weights)
        predictions.append(predicted_house)
    
    results_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })
    
    results_df.to_csv('output/houses.csv', index=False)
    print(f"Predictions saved to 'output/houses.csv'")
    
    return predictions

if __name__ == "__main__":
    predictions = predict_all_students()