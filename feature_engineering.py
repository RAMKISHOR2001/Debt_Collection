import pandas as pd

def engineer_features(df):
    df['debt_to_income_ratio'] = df['debt_amount'] / df['income']
    df['default_rate'] = df['num_previous_defaults'] / (df['num_previous_loans'] + 1)
    
    return df

def get_features():
    return ['debt_amount', 'days_overdue', 'credit_score', 'age', 'income', 
            'num_previous_loans', 'num_previous_defaults', 'communication_preference',
            'days_since_last_contact', 'debt_to_income_ratio', 'default_rate']

if __name__ == "__main__":
    df = pd.read_csv('preprocessed_data.csv')
    df_engineered = engineer_features(df)
    df_engineered.to_csv('C:/Users/Ramkishor/Downloads/New folder/CogniHack/engineered_data.csv', index=False)
    print("Features engineered and saved to 'engineered_data.csv'")