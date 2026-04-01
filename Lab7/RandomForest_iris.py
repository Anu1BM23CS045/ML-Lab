import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("iris.csv")

# Features and target
X = data.drop("species", axis=1)   # change column name if needed
y = data["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# 1. Random Forest with default n_estimators = 10
# -------------------------------
rf_default = RandomForestClassifier(n_estimators=10, random_state=42)
rf_default.fit(X_train, y_train)

y_pred_default = rf_default.predict(X_test)
default_score = accuracy_score(y_test, y_pred_default)

print("Accuracy with default n_estimators (10):", default_score)



# Variables to store best results
best_score = 0
best_n = 0
best_model = None

# Try different number of trees
for n in range(1, 101):
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    if score > best_score:
        best_score = score
        best_n = n
        best_model = rf

# Final predictions using best model
y_best_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_best_pred)

# Output results
print("Best Accuracy:", best_score)
print("Best n_estimators (number of trees):", best_n)
print("\nConfusion Matrix:\n", cm)
