# =====================================
# Import Libraries
# =====================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# =====================================
# Load Data
# =====================================
diabetes_df = pd.read_csv("diabetes.csv")
adult_df = pd.read_csv("adult.csv")

# =====================================
# 🔹 1. DATA CLEANING
# =====================================

# ---------- DIABETES DATASET ----------
print("Diabetes Dataset Info:\n", diabetes_df.info())

# Drop unnecessary columns
diabetes_df.drop(['ID', 'No_Pation'], axis=1, inplace=True)

# Convert categorical (Gender, CLASS)
le = LabelEncoder()
diabetes_df['Gender'] = le.fit_transform(diabetes_df['Gender'])   # F/M → 0/1
diabetes_df['CLASS'] = le.fit_transform(diabetes_df['CLASS'])     # N/P → 0/1

# Check missing values
print("\nMissing values:\n", diabetes_df.isnull().sum())

# Fill missing values with mean (if any)
diabetes_df.fillna(diabetes_df.mean(), inplace=True)

# ---------- ADULT DATASET ----------
print("\nAdult Dataset Info:\n", adult_df.info())

# Replace '?' with NaN
adult_df.replace('?', np.nan, inplace=True)

# Drop missing rows
adult_df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in adult_df.select_dtypes(include='object').columns:
    adult_df[col] = le.fit_transform(adult_df[col])

# =====================================
# 🔹 2. OUTLIER HANDLING (IQR METHOD)
# =====================================
def remove_outliers(df):
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

diabetes_df = remove_outliers(diabetes_df)
adult_df = remove_outliers(adult_df)

# =====================================
# 🔹 3. DATA TRANSFORMATION
# =====================================

# -------- Min-Max Scaling --------
minmax = MinMaxScaler()

diabetes_minmax = pd.DataFrame(
    minmax.fit_transform(diabetes_df),
    columns=diabetes_df.columns
)

adult_minmax = pd.DataFrame(
    minmax.fit_transform(adult_df),
    columns=adult_df.columns
)

# -------- Standard Scaling --------
scaler = StandardScaler()

diabetes_scaled = pd.DataFrame(
    scaler.fit_transform(diabetes_df),
    columns=diabetes_df.columns
)

adult_scaled = pd.DataFrame(
    scaler.fit_transform(adult_df),
    columns=adult_df.columns
)

# =====================================
# OUTPUT
# =====================================
print("\nProcessed Diabetes Data (Standard Scaled):")
print(diabetes_scaled.head())

print("\nProcessed Adult Data (Standard Scaled):")
print(adult_scaled.head())
