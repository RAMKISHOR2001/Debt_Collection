import pandas as pd
import numpy as np
import joblib

def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def optimize_collection_strategy(model, customer_data):
    payment_prob = model.predict_proba(customer_data)[:, 1]
    
    def get_strategy(prob):
        if prob < 0.3:
            return "Aggressive collection"
        elif prob < 0.7:
            return "Standard follow-up"
        else:
            return "Soft reminder"
    
    strategies = [get_strategy(prob) for prob in payment_prob]
    return strategies

def apply_optimization(df, model, scaler, features):
    X = df[features]
    X_scaled = scaler.transform(X)
    optimized_strategies = optimize_collection_strategy(model, X_scaled)
    
    df_results = pd.DataFrame({
        'customer_id': df['customer_id'],
        'debt_amount': df['debt_amount'],
        'payment_probability': model.predict_proba(X_scaled)[:, 1],
        'recommended_strategy': optimized_strategies
    })
    
    return df_results

if __name__ == "__main__":
    from feature_engineering import get_features
    
    model, scaler = load_model_and_scaler('xgboost_model.joblib', 'scaler.joblib')
    df = pd.read_csv('engineered_data.csv')
    features = get_features()
    
    results = apply_optimization(df, model, scaler, features)
    results.to_csv('C:/Users/Ramkishor/Downloads/New folder/CogniHack/optimization_results.csv', index=False)
    print("Optimization results saved to 'optimization_results.csv'")