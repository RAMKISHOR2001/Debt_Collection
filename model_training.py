import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
from feature_engineering import get_features

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    features = get_features()
    X = df[features]
    y = (df['payment_probability'] > 0.5).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    print(f"{model_name} Performance:")
    print(classification_report(y, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y, y_prob):.4f}\n")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data('engineered_data.csv')
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    rf_model = train_random_forest(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)
    
    evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    
    # Save models and scaler
    joblib.dump(rf_model, 'random_forest_model.joblib')
    joblib.dump(xgb_model, 'xgboost_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print("Models and scaler saved.")