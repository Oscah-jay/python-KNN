import csv
import numpy as np
from collections import Counter

# Function to load CSV data


def load_csv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)  # Skip header
        for row in csv_reader:
            try:
                # Convert all values to float
                data.append([float(i) for i in row])
            except ValueError as e:
                print(f"Error in row: {row} - {e}")
    return np.array(data)  # Convert to numpy array for faster computations

# Function to calculate Euclidean distance


def euclidean_distance(train_row, test_row):
    return np.sqrt(np.sum((train_row - test_row) ** 2))

# Function to find K-nearest neighbors


def get_k_nearest_neighbors(train_data, test_row, k):
    distances = np.sqrt(
        np.sum((train_data[:, 1:] - test_row[1:]) ** 2, axis=1))
    nearest_indices = np.argsort(distances)[:k]
    return train_data[nearest_indices, 0]

# Function to predict class based on majority vote


def predict_classification(neighbors):
    return Counter(neighbors).most_common(1)[0][0]

# Function to evaluate the model's accuracy


def evaluate_accuracy(test_data, predictions):
    correct = np.sum(test_data[:, 0] == predictions)
    return correct / len(test_data)

# Main KNN function


def knn(train_data, test_data, k):
    predictions = []
    for test_row in test_data:
        neighbors = get_k_nearest_neighbors(train_data, test_row, k)
        prediction = predict_classification(neighbors)
        predictions.append(prediction)
    accuracy = evaluate_accuracy(test_data, np.array(predictions))
    return accuracy, predictions


# Load datasets
train_data = load_csv('mnist_train.csv')
test_data = load_csv('mnist_test.csv')

# Define the value of K
k = 3

# Run KNN and get accuracy
accuracy, predictions = knn(train_data, test_data, k)

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Number of predictions: {len(predictions)}")
