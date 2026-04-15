import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# Ensure 'income.csv' is in the same directory
try:
    df = pd.read_csv('/content/income.csv')
except FileNotFoundError:
    print("Error: income.csv not found. Please ensure the file exists.")
    exit()

# 2. Preprocessing
# AdaBoost requires numerical data. We encode categorical variables.
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Splitting features (X) and target (y)
# Assuming the last column is the target 'income'
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Change 'test_test_split' to 'test_size'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Task i: Build AdaBoost with n_estimators=10 ---
model_10 = AdaBoostClassifier(n_estimators=10, random_state=42)
model_10.fit(X_train, y_train)
y_pred_10 = model_10.predict(X_test)
score_10 = accuracy_score(y_test, y_pred_10)

print(f"Prediction score with 10 estimators: {score_10:.4f}")

# --- Task ii: Fine-tune by changing the number of trees ---
results = {}
estimator_range = [10, 50, 100, 200, 500, 1000]

print("\nFine-tuning model...")
for n in estimator_range:
    model = AdaBoostClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results[n] = score
    print(f"Trees: {n} | Accuracy: {score:.4f}")

# Identify the best score and best number of trees
best_n = max(results, key=results.get)
best_score = results[best_n]

print("-" * 30)
print(f"Best Score: {best_score:.4f}")
print(f"Optimal number of trees: {best_n}")

# Optional: Visualize the tuning process
plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='--')
plt.title('AdaBoost Performance vs Number of Trees')
plt.xlabel('Number of Estimators (Trees)')
plt.ylabel('Accuracy Score')
plt.grid(True)
plt.show()
