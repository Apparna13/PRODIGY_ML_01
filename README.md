import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load train dataset
train_data = pd.read_csv('/content/train.csv')

# Feature extraction and preprocessing
train_data['TotalBaths'] = train_data['FullBath'] + 0.5 * train_data['HalfBath']
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBaths']
X = train_data[features]
y = train_data['SalePrice']

# Handle missing values (fill with median)
X = X.fillna(X.median())
y = y.fillna(y.median())

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Predict on a new test dataset
# Load test dataset
test_data = pd.read_csv('/content/test.csv')
test_data['TotalBaths'] = test_data['FullBath'] + 0.5 * test_data['HalfBath']
X_new = test_data[features].fillna(X.median())

# Predict prices
test_data['PredictedPrice'] = model.predict(X_new)

# Save predictions
test_data[['Id', 'PredictedPrice']].to_csv('predictions.csv', index=False)
