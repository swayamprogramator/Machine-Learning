import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from Kaggle
# Ensure you have the dataset downloaded as 'house_prices.csv'
df = pd.read_csv('train.csv')

# Assume 'SqFt' is the feature and 'Price' is the target
X = df['LotArea'].values  # Feature (Square Footage)
y = df['SalePrice'].values  # Target (House Price)

# Normalize data for better convergence
X_mean, X_std = np.mean(X), np.std(X)
y_mean, y_std = np.mean(y), np.std(y)
X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

# Initialize parameters
w = 0  # Weight
b = 0  # Bias
alpha = 0.01  # Learning rate
iterations = 500  # Number of iterations


# Gradient Descent Algorithm
def compute_cost(X, y, w, b):
    m = len(y)
    cost = np.sum((w * X + b - y) ** 2) / (2 * m)
    return cost


def gradient_descent(X, y, w, b, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        dw = np.sum((w * X + b - y) * X) / m
        db = np.sum(w * X + b - y) / m

        w -= alpha * dw
        b -= alpha * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b, cost_history


# Train the model
w, b, cost_history = gradient_descent(X, y, w, b, alpha, iterations)
print("w=",w,"b=",b,"a=",alpha)

# Predict function
def predict(sq_ft):
    x_norm = (sq_ft - X_mean) / X_std  # Normalize input
    y_pred_norm = w * x_norm + b
    y_pred = y_pred_norm * y_std + y_mean  # De-normalize output
    return y_pred


# Test the prediction
sq_ft = 2000
predicted_price = predict(sq_ft)
print(f"Predicted house price for {sq_ft} sq. ft: ${predicted_price:.2f}")


# Plot cost function convergence
plt.plot(range(iterations), cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()



