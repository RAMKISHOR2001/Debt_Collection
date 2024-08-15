import pandas as pd
from datetime import datetime

def preprocess_data(df):
    # Convert last_contact_date to datetime
    df['last_contact_date'] = pd.to_datetime(df['last_contact_date'])
    
    # Calculate days since last contact
    df['days_since_last_contact'] = (datetime.now() - df['last_contact_date']).dt.days
    
    # Convert communication_preference to numerical
    df['communication_preference'] = df['communication_preference'].map({'email': 0, 'phone': 1, 'mail': 2})
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('debt_collection_data.csv')
    df_preprocessed = preprocess_data(df)
    df_preprocessed.to_csv('C:/Users/Ramkishor/Downloads/New folder/CogniHack/preprocessed_data.csv', index=False)
    print("Data preprocessed and saved to 'preprocessed_data.csv'")