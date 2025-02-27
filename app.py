import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import xgboost as xgb
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load trained models
custom_objects = {"mse": MeanSquaredError()}
ann_model = load_model(r"D:\Research\best_ann.h5", custom_objects=custom_objects)
lstm_model = load_model(r"D:\Research\best_lstm.h5", custom_objects=custom_objects)

with open(r"D:\Research\best_xgb.pkl", "rb") as file:
    xgb_model = pickle.load(file)

# Load dataset
data = pd.read_csv(r"D:\Research\Project\Bengaluru_House_Data.csv")
feature_columns = ['total_sqft', 'bath', 'balcony']

# Function to clean and preprocess 'total_sqft' column
def convert_sqft(sqft):
    try:
        if '-' in str(sqft):
            sqft = (float(sqft.split('-')[0]) + float(sqft.split('-')[1])) / 2
        return float(sqft)
    except:
        return np.nan

# Preprocess dataset
data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
data.dropna(inplace=True)

# Fit StandardScaler
scaler = StandardScaler()
scaler.fit(data[feature_columns])

# Streamlit UI
st.title("🏡 Bangalore House Price Prediction")
st.sidebar.header("🔢 Predicted House Prices")

# User input fields
total_sqft = st.sidebar.text_input("Total Sqft (e.g., 1200 or 1000 - 1500)", "1200")
bath = st.sidebar.slider("Number of Bathrooms", 1, 10, 2)
balcony = st.sidebar.slider("Number of Balconies", 0, 5, 1)

# Convert and scale input
input_data = pd.DataFrame([[convert_sqft(total_sqft), bath, balcony]], columns=feature_columns)
input_data.dropna(inplace=True)

if not input_data.empty:
    X_scaled = scaler.transform(input_data)

    # Predictions
    xgb_pred = xgb_model.predict(X_scaled)
    ann_pred = ann_model.predict(X_scaled).flatten()
    lstm_pred = lstm_model.predict(X_scaled).flatten()
    hybrid_pred = (xgb_pred + ann_pred + lstm_pred) / 3

    # Display Predictions
    st.subheader("🏡 Bangalore House Price Prediction")
    st.write(f"🏆 *XGBoost Prediction:* ₹ {xgb_pred[0]:,.2f}")
    st.write(f"🤖 *ANN Prediction:* ₹ {ann_pred[0]:,.2f}")
    st.write(f"📈 *LSTM Prediction:* ₹ {lstm_pred[0]:,.2f}")
    st.write(f"💡 *Hybrid Model Prediction:* ₹ {hybrid_pred[0]:,.2f}")

    # Visualization - Model Comparison
    st.subheader("📊 Model Comparison")
    fig, ax = plt.subplots()
    models = ["XGBoost", "ANN", "LSTM", "Hybrid"]
    predictions = [xgb_pred[0], ann_pred[0], lstm_pred[0], hybrid_pred[0]]
    sns.barplot(x=models, y=predictions, palette="viridis", ax=ax)
    ax.set_ylabel("Price in ₹")
    st.pyplot(fig)

    # SHAP Explainability for XGBoost
    st.subheader("🔍 Feature Importance (SHAP)")
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_scaled)

    # SHAP Bar Plot
    # SHAP Feature Importance Bar Plot (Corrected)
    st.subheader("📌 Feature Importance (Bar Plot)")
    shap_values_mean = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": shap_values_mean})
    shap_importance_df.sort_values(by="Importance", ascending=True, inplace=True)  # Sorted for better visualization

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y=shap_importance_df["Feature"], x=shap_importance_df["Importance"], palette="coolwarm", ax=ax)
    ax.set_xlabel("SHAP Importance Value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    st.pyplot(fig)


    # SHAP Summary Plot with Improved Labels and Layout
    st.subheader("📊 SHAP Summary Plot")
    shap_fig, shap_ax = plt.subplots(figsize=(10, 7))

    # Generate SHAP summary plot
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_columns, show=False)

    # Adjust labels for better readability
    shap_ax.set_xlabel("SHAP Value Impact on Model Output", fontsize=12)
    shap_ax.set_ylabel("Features", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)  # Add light grid for readability
    plt.tight_layout()

    # Display the improved plot
    st.pyplot(shap_fig)


# Dataset Exploration
st.subheader("📊 Data Insights")

# Histogram for each feature
st.subheader("📌 Feature Distributions")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(data["total_sqft"], kde=True, bins=30, ax=ax[0], color="blue")
ax[0].set_title("Total Sqft Distribution")
sns.histplot(data["bath"], kde=True, bins=10, ax=ax[1], color="green")
ax[1].set_title("Bathroom Distribution")
sns.histplot(data["balcony"], kde=True, bins=5, ax=ax[2], color="red")
ax[2].set_title("Balcony Distribution")
st.pyplot(fig)

# Pairplot
st.subheader("📈 Feature Relationships")
fig = sns.pairplot(data[feature_columns])
st.pyplot(fig)
