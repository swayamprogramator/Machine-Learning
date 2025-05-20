import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split dataset manually
def train_test_split_manual(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

X = df.drop(columns=['target']).values
y = df['target'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)

# Normalize features
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Add bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Initialize weights
np.random.seed(42)
theta = np.random.randn(X_train.shape[1], 1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
learning_rate = 0.01
epochs = 1000
m = len(y_train)

for epoch in range(epochs):
    z = np.dot(X_train, theta)
    h = sigmoid(z)
    gradient = np.dot(X_train.T, (h - y_train)) / m
    theta -= learning_rate * gradient

# Predictions
def predict(X, theta):
    return (sigmoid(np.dot(X, theta)) >= 0.5).astype(int)

y_pred = predict(X_test, theta)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.4f}')
