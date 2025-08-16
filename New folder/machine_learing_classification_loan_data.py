# ===========================
# 1️⃣ Import Libraries
# ===========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# ===========================
# 2️⃣ Load Dataset
# ===========================
df = pd.read_csv("C:\Users\NS COMPUTERS\Documents\GitHub\AIcourse\New folder\loan_data.csv")
print("Shape of dataset:", df.shape)
print("\nPreview:\n", df.head())

# ===========================
# 3️⃣ Data Overview
# ===========================
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# Purpose is categorical → Encode
df['purpose'] = df['purpose'].astype('category').cat.codes

# Target variable (assuming 'paid.back.loan' exists as per Lending Club dataset)
target_col = 'paid.back.loan'  # adjust if name differs
if target_col not in df.columns:
    target_col = df.columns[-1]  # fallback if column name varies

X = df.drop(columns=[target_col])
y = df[target_col]

# ===========================
# 4️⃣ Train-Test Split & Scaling
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# 5️⃣ Exploratory Data Analysis
# ===========================
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.countplot(x=target_col, data=df)
plt.title("Class Distribution")
plt.show()

# ===========================
# 6️⃣ Model Training
# ===========================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# ===========================
# 7️⃣ Evaluation
# ===========================
y_pred = rf.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC (if binary classification)
try:
    y_proba = rf.predict_proba(X_test_scaled)[:,1]
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
except:
    print("ROC-AUC skipped (non-binary target)")