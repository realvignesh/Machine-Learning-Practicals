# 1. IMPORTING LIBRARIES AND LOADING DATASET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print()
print("1. LOADING THE DATASET".center(60), "\n")
df = pd.read_csv("Datasets\\P2-USA-Housing.csv")

print("Example Records From The Dataset")
print(df.head(5))
print("\n")

# ======================================================================================================================


# 2. DATA PREPROCESSING
print("2. DATA PREPROCESSING".center(60), "\n")

# Check for missing values
print("Checking Missing Values:")
print(df.isnull().sum())
print()

print("Selecting the Label and Features")
print("Chosen Label - 'price'")
y = df.iloc[:, 1]
print(y.head(5), "\n")

print("Chosen Features - 'bedrooms' to 'condition'")
X = df.iloc[:, 2:10]
print(X.head(5), "\n")

print("Splitting the Training and Testing Data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print('Shape of Train Set:', X_train.shape, y_train.shape)
print('Shape of Test Set:', X_test.shape, y_test.shape)
print("\n")

# ======================================================================================================================


# 3. BUILDING AND EVALUATING LINEAR REGRESSION MODEL
print("3. BUILDING AND EVALUATING LINEAR REGRESSION MODEL".center(60), "\n")

# Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)
print("\n")

# ======================================================================================================================


# 4. PREDICTIONS AND VISUALIZATION
print("4. PREDICTIONS AND VISUALIZATION".center(60), "\n")

# Scatter plot to visualize the predictions against actual prices
plt.scatter(y_test, y_pred)
m, b = np.polyfit(y_pred, y_test, 1)
plt.plot(y_pred, m*y_pred + b, color='black')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

# Residual plot to check the model's performance
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Using the trained model to make predictions on new data.
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price[0])
