# =====================================
# Import Libraries
# =====================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("iris.csv")

# =====================================
# Data Preparation
# Convert species into binary classification
# (Iris-setosa = 0, Others = 1)
# =====================================
df['species'] = df['species'].apply(lambda x: 0 if x == 'Iris-setosa' else 1)

# Features and Target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# Train Logistic Regression Model
# =====================================
model = LogisticRegression()
model.fit(X_train, y_train)

# =====================================
# Predictions
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
