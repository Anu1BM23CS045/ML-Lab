# =====================================
# Import Libraries
# =====================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("iris.csv")

# =====================================
# 🔹 1. LINEAR REGRESSION (Single Feature)
# Predict petal_width using petal_length
# =====================================
X_linear = df[['petal_length']]   # Independent variable
y_linear = df['petal_width']      # Dependent variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# Evaluation
print("---- Linear Regression ----")
print("Coefficient:", lr_model.coef_)
print("Intercept:", lr_model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# =====================================
# 🔹 2. MULTIPLE LINEAR REGRESSION
# Predict petal_width using multiple features
# =====================================
X_multi = df[['sepal_length', 'sepal_width', 'petal_length']]
y_multi = df['petal_width']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train model
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# Predictions
y_pred_multi = mlr_model.predict(X_test)

# Evaluation
print("\n---- Multiple Linear Regression ----")
print("Coefficients:", mlr_model.coef_)
print("Intercept:", mlr_model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_multi))
print("R2 Score:", r2_score(y_test, y_pred_multi))
