# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np

# Load dataset
data = pd.read_csv("/petrol_consumption.csv")

# Define features (X) and target (y)
X = data[['Petrol_tax','Average_income','Paved_Highways',
          'Population_Driver_licence(%)']]

y = data['Petrol_Consumption']

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Regression Tree model
model = DecisionTreeRegressor()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Calculate error metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display results
print("Accuracy Score:", accuracy)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
