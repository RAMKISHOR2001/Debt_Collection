import pandas as pd
from dataGen import generate_data
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features, get_features
from model_training import load_and_prepare_data, scale_data, train_random_forest, train_xgboost, evaluate_model
from optimization_strat import load_model_and_scaler, apply_optimization

def main():
    # Generate data
    print("Generating data...")
    df = generate_data()
    df.to_csv('debt_collection_data.csv', index=False)
    
    # Preprocess data
    print("Preprocessing data...")
    df_preprocessed = preprocess_data(df)
    
    # Engineer features
    print("Engineering features...")
    df_engineered = engineer_features(df_preprocessed)
    df_engineered.to_csv('engineered_data.csv', index=False)
    
    # Train models
    print("Training models...")
    X_train, X_test, y_train, y_test = load_and_prepare_data('engineered_data.csv')
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    rf_model = train_random_forest(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)
    
    evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    
    # Apply optimization strategy
    print("Applying optimization strategy...")
    features = get_features()
    results = apply_optimization(df_engineered, xgb_model, scaler, features)
    
    # Save results
    results.to_csv('final_results.csv', index=False)
    print("Final results saved to 'final_results.csv'")

if __name__ == "__main__":
    main()