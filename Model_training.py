### Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

### Step 2: Load Dataset
data = pd.read_csv(r"D:\Research\Project\Bengaluru_House_Data.csv")
data = data[['total_sqft', 'bath', 'balcony', 'price']]  # Selecting relevant features
data.dropna(inplace=True)
def convert_sqft_to_num(sqft):
    try:
        # If the value contains a range (e.g., "2100 - 2850"), take the average
        if '-' in sqft:
            sqft_vals = list(map(float, sqft.split('-')))
            return (sqft_vals[0] + sqft_vals[1]) / 2  
        return float(sqft)  # Convert single values directly to float
    except:
        return None  # Return None for invalid entries
data['total_sqft'] = data['total_sqft'].astype(str).apply(convert_sqft_to_num)
data.dropna(inplace=True)  # Drop rows where conversion failed

X = data.drop(columns=['price'])  # Features
y = data['price']  # Target Variable

### Step 3: Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

### Save Scaler for Deployment
pd.DataFrame(X_scaled, columns=X.columns).to_csv("scaler_data.csv", index=False)

### Step 4: Train XGBoost with GridSearchCV
param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_
xgb_pred = best_xgb.predict(X_test)

### Save XGBoost Model
pickle.dump(best_xgb, open("best_xgb.pkl", "wb"))

### Step 5: Train ANN Model
def build_ann(hp):
    model = Sequential([
        Dense(hp.Int('units', 32, 128, step=32), activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

tuner = kt.RandomSearch(build_ann, objective='val_loss', max_trials=5)
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
best_ann = tuner.get_best_models(num_models=1)[0]
ann_pred = best_ann.predict(X_test).flatten()

### Save ANN Model
best_ann.save("best_ann.h5")

### Step 6: Train LSTM Model
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(50, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=16, validation_split=0.2)
lstm_pred = lstm_model.predict(X_test_lstm).flatten()

### Save LSTM Model
lstm_model.save("best_lstm.h5")

### Step 7: Hybrid Model
hybrid_pred = (xgb_pred + ann_pred + lstm_pred) / 3

### Step 8: Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{model_name} Performance:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAPE:", mape)
    print("R² Score:", r2_score(y_true, y_pred))
    print("\n")

evaluate_model(y_test, xgb_pred, "XGBoost (ML)")
evaluate_model(y_test, ann_pred, "ANN (DL)")
evaluate_model(y_test, lstm_pred, "LSTM (DL)")
evaluate_model(y_test, hybrid_pred, "Hybrid Model (ML+DL)")

### Step 9: SHAP for Explainability
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=["total_sqft", "bath", "balcony"])
