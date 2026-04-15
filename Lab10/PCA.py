import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# 1. Load Dataset
try:
    df = pd.read_csv('/content/heart (1).csv')
except FileNotFoundError:
    print("Error: heart.csv not found.")
    exit()

# 2. Preprocessing: Encoding
# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Example: Using Label Encoding for Binary and One-Hot for multi-category
# (Adjust logic based on your specific 'heart.csv' columns)
le = LabelEncoder()
for col in categorical_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# 3. Split Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build and Evaluate Models (Before PCA)
models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("--- Accuracy BEFORE PCA ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"{name}: {acc:.4f}")

# 6. Apply PCA
# Reducing to a smaller number of components (e.g., keeping 95% of variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nReduced feature count from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]} using PCA.")

# 7. Evaluate Models (After PCA)
print("\n--- Accuracy AFTER PCA ---")
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_pca))
    print(f"{name}: {acc:.4f}")
