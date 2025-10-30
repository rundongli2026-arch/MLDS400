# titanic_model.py
# Author: Rundong Li
# Description: Logistic Regression model for Titanic survival prediction
# This script loads Titanic train and test data, cleans it, trains a model, and prints outputs.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("Starting Titanic survival prediction script...")

# ===============================
# 1. Load Data
# ===============================
train = pd.read_csv("src/data/train.csv")
test = pd.read_csv("src/data/test.csv")

print("\nData loaded successfully.")
print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# ===============================
# 2. Basic Data Cleaning
# ===============================
print("\nCleaning data...")

# Fill missing Age and Fare values with median
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# Fill missing Embarked values with mode
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)

# Convert categorical features
le = LabelEncoder()
for col in ["Sex", "Embarked"]:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])

print("Missing values handled and categorical variables encoded.")

# ===============================
# 3. Feature Selection
# ===============================
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = train[features]
y = train[target]

print("\nFeatures used:", features)

# ===============================
# 4. Train/Test Split and Model Training
# ===============================
print("\nTraining logistic regression model...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training complete.")

# ===============================
# 5. Evaluate on Training Data
# ===============================
train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f"Training accuracy: {train_acc:.4f}")

# ===============================
# 6. Evaluate on Validation Data
# ===============================
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.4f}")

# ===============================
# 7. Predict on Test Data
# ===============================
print("\nPredicting on test dataset...")
test_pred = model.predict(test[features])

# Save results
output = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})
output.to_csv("src/data/predictions.csv", index=False)
print("Predictions saved to src/data/predictions.csv")

print("\nTitanic model run complete.")
