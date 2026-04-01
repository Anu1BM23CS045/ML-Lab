# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("letter-recognition.csv")

# Features and target
X = df.drop("letter", axis=1)
y = df["letter"]

# Encode target labels (A-Z → 0-25)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Binarize labels for ROC (multi-class)
y_bin = label_binarize(y_encoded, classes=np.arange(26))

# Train-test split (80/20)
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y_encoded, y_bin, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature scaling (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM (RBF kernel with probability for ROC)
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------- ROC Curve & AUC --------

# Get probability scores
y_score = svm_model.predict_proba(X_test)

# Compute ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(26):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"],
         label="Micro-average ROC (AUC = %0.2f)" % roc_auc["micro"],
         color='deeppink', linestyle=':', linewidth=3)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Letter Recognition (SVM)")
plt.legend(loc="lower right")
plt.show()

# Print AUC score
print("AUC Score (Micro-average):", roc_auc["micro"])
