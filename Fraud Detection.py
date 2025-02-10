#### 1) Load the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\Samruddhi Yadav\Documents\Resume Project\Fraud detection\archive (5)\creditcard.csv"
df = pd.read_csv(file_path)

# Display basic information
print(df.info())
print(df.head())

# Check missing values
print(df.isnull().sum())  # No missing values expected

# Check class distribution
print(df["Class"].value_counts())
sns.countplot(x="Class", data=df)
plt.show()

#### 2) Feature Engineering
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

df = df.drop(columns=["Time"])

####  3) Handle Imbalanced Data (SMOTE Oversampling)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())

#### 4) Train Machine Learning Model

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize and train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#### 5) Hyperparameter Tunning
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss"), param_grid=params, cv=3)
grid_search.fit(X_train_resampled, y_train_resampled)

print("Best Parameters:", grid_search.best_params_)

import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit UI
st.title("💳 Fraud Detection System")
st.write("Upload a transaction dataset to predict fraud.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data (avoid using chunks unless absolutely necessary)
    df = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.write("### Dataset Preview")
    st.write(df.head())

    # Ensure dataset contains correct columns
    if "Class" not in df.columns:
        st.error("❌ Invalid dataset! The dataset must contain a 'Class' column (0 for legit, 1 for fraud).")
    else:
        # Splitting Features and Labels
        X = df.drop(columns=["Class"])  # Features
        y = df["Class"]  # Target variable

        # Splitting dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the XGBoost model (Cache to avoid reloading on UI interactions)
        @st.cache_resource
        def train_model():
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)
            return model

        model = train_model()

        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Model accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"✅ **Model Accuracy on Test Data: {accuracy:.2f}**")

        # Make predictions on full dataset
        df["Prediction"] = model.predict(X)
        df["Prediction"] = df["Prediction"].map({0: "Legit", 1: "Fraudulent"})

        # Show results
        st.write("### Prediction Results")
        st.write(df.head(20))  # Show first 20 predictions

        # Fraud Transaction Count
        fraud_count = (df["Prediction"] == "Fraudulent").sum()
        st.write(f"🔴 **Detected Fraudulent Transactions: {fraud_count}**")

        # Download Results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "fraud_predictions.csv", "text/csv")
