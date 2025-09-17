import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt   # <-- add this
import seaborn as sns            # <-- add this

# ...existing code...

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    st.error("Model or scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory as this script.")
    st.stop()

def predict_diabetes(features):
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

st.sidebar.title("Diabetes Prediction App")
option = st.sidebar.radio(
    "Select Page",
    ["Data Exploration", "Visualization", "Prediction", "Model Performance"]
)
# ...existing code...

# Example: Load your data (adjust path if needed)
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

if option == "Data Exploration":
    st.title("Data Exploration")
    df = load_data()
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Sample Data:")
    st.dataframe(df.head())

elif option == "Visualization":
    st.title("Visualization")
    df = load_data()
    st.write("Correlation Heatmap:")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif option == "Prediction":
    st.title("Prediction")
    df = load_data()
    st.write("Enter values to predict diabetes:")
    input_data = []
    for col in df.columns[:-1]:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data.append(val)
    if st.button("Predict"):
        result = predict_diabetes(input_data)
        st.success(f"Prediction: {result}")

elif option == "Model Performance":
    st.title("Model Performance")
    df = load_data()
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {acc:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)