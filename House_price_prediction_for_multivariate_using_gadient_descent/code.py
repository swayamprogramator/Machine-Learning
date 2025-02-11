import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (Ensure your CSV file is correct)
df = pd.read_csv('train.csv')

# Select features and target variable
features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
target = 'SalePrice'


# Extract features (X) and target (y)
X = df[features].values  # Shape (m, 4)
y = df[target].values.reshape(-1, 1)  # Shape (m, 1)

# Normalize X and y (Feature Scaling)
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
y_mean, y_std = y.mean(), y.std()
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

# Add Bias Term (X0 = 1)
m, n = X.shape
X = np.c_[np.ones((m, 1)), X]  # Shape becomes (m, 5) (Adding bias column)

# Initialize Parameters
W = np.zeros((n + 1, 1))  # (5, 1) including bias
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations


# Cost Function (Mean Squared Error)
def compute_cost(X, y, W):
    m = len(y)
    predictions = X @ W  # Matrix multiplication
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# Gradient Descent Algorithm
def gradient_descent(X, y, W, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        gradients = (1 / m) * X.T @ (X @ W - y)  # Compute gradients
        W -= alpha * gradients  # Update weights

        cost = compute_cost(X, y, W)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return W, cost_history


# Train the model
W, cost_history = gradient_descent(X, y, W, alpha, iterations)

# Plot cost function convergence
plt.plot(range(iterations), cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()


# Prediction Function
def predict(features):
    features = np.array(features).reshape(1, -1)  # Convert input to numpy array
    features = (features - X_mean) / X_std  # Normalize
    features = np.c_[np.ones((1, 1)), features]  # Add bias term
    prediction_norm = features @ W  # Compute prediction
    prediction = prediction_norm * y_std + y_mean  # De-normalize
    return prediction[0, 0]


# Test the prediction
test_features = [2000, 3, 2, 2015]  # Example: 2000 SqFt, 3 Beds, 2 Baths, Built 2015
predicted_price = predict(test_features)
print(f"Predicted house price: ${predicted_price:.2f}")
