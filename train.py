import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

print("Loading dataset...")
df = pd.read_csv("loan_data.csv")

# Drop unique identifier if present
if "Loan_ID" in df.columns:
  df = df.drop(columns=["Loan_ID"])

# Handle missing values and convert numerical columns safely
numerical_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
for col in numerical_cols:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Handle missing values for categorical columns
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status",
]
for col in categorical_cols:
  if col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Clean the 'Dependents' column (replace '3+' with '3' and make numeric)
if "Dependents" in df.columns:
  df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
  df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0)

# Encode categorical text columns into numbers
le = LabelEncoder()
encode_cols = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status",
]
for col in encode_cols:
  if col in df.columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Define Features (X) and Target (y)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
}

best_acc = 0
best_model = None
best_name = ""

print("\n--- Model Evaluation ---")
for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  print(f"{name} Accuracy: {acc * 100:.2f}%")
  if acc > best_acc:
    best_acc = acc
    best_model = model
    best_name = name

print(f"\nBest Model Selected: {best_name} ({best_acc * 100:.2f}% accuracy)")

# Save the Best Model
joblib.dump(best_model, "loan_model.pkl")
print("Model saved successfully as 'loan_model.pkl'!")