import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (Ensure the file 'house_prices.csv' is in the same directory)
df = pd.read_csv('train.csv')

# Extract feature (SqFt) and target (Price)
X = df['LotArea'].values
y = df['SalePrice'].values

# Compute mean values
X_mean = np.mean(X)
y_mean = np.mean(y)

# Compute variance and covariance
variance_X = np.var(X, ddof=0)  # Population variance
cov_XY = np.cov(X, y, bias=True)[0, 1]  # Population covariance

# Compute w and b
w = cov_XY / variance_X
b = y_mean - w * X_mean

print(f"Calculated parameters: w = {w:.4f}, b = {b:.4f}")

# Function to make predictions
def predict(X_new):
    return w * X_new + b

# Plot regression line
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, predict(X), color='red', label="Regression Line")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.title("House Price Prediction using Variance & Covariance")
plt.legend()
plt.show()

# Predict for a new house size
sq_ft = 1800
predicted_price = predict(sq_ft)
print(f"Predicted price for {sq_ft} sq. ft: ${predicted_price:.2f}")
