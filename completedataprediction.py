# Multiple Linear Regression for Groundwater Level Prediction

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r'D:\M.Tech DS\AIML application\Assign\groundwater_all_cleaned_encoded.csv')

# Specify target and features
target = 'dataValue'  # Assuming 'dataValue' is the column for groundwater level
# Use all columns except target as features
features = [col for col in data.columns if col != target]

X = data[features].fillna(data[features].mean())
y = data[target].fillna(data[target].mean())

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Multiple Linear Regression model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# Predict on test data
y_pred = mlr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = mlr_model.score(X_test, y_test)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
print(f"Accuracy: {accuracy * 100}")  # in % * 100

# Optional: visualize predicted vs actual groundwater levels
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # reference line
plt.xlabel('Actual Groundwater Level')
plt.ylabel('Predicted Groundwater Level')
plt.title('MLR: Actual vs Predicted Groundwater Level')
plt.show()

# Save the trained model
import joblib
joblib.dump(mlr_model, r'D:\M.Tech DS\AIML application\Assign\mlr_groundwater_model.pkl')
print("Model saved successfully!")
