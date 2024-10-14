import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import functions from the previous implementation
from knn_mnist import load_csv, predict

# Load the MNIST data
print("Loading MNIST data...")
mnist_data = load_csv('mnist_train.csv')
X, y = mnist_data[:, 1:], mnist_data[:, 0]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the range of K values to test
ks = range(1, 10)
accuracies = []

print("Testing different K values...")
for k in ks:
    print(f"Testing k={k}...")
    y_pred = predict(X_train_scaled, y_train, X_test_scaled, k)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy * 100)
    print(f"k={k}, Accuracy: {accuracy * 100:.2f}%")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(ks, accuracies, marker='o')
plt.xlabel('K')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs K for KNN on MNIST')
plt.xticks(ks)
plt.grid(True)
plt.show()

# Find the best K
best_k = ks[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"\nBest K: {best_k}")
print(f"Best Accuracy: {best_accuracy:.2f}%")
