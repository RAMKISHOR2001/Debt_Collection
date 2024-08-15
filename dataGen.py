import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(n_records=10000, seed=42):
    np.random.seed(seed)
    
    data = {
        'customer_id': range(1, n_records + 1),
        'debt_amount': np.random.uniform(100, 10000, n_records).round(2),
        'days_overdue': np.random.randint(1, 365, n_records),
        'credit_score': np.random.randint(300, 850, n_records),
        'age': np.random.randint(18, 80, n_records),
        'income': np.random.uniform(20000, 200000, n_records).round(2),
        'num_previous_loans': np.random.randint(0, 10, n_records),
        'num_previous_defaults': np.random.randint(0, 5, n_records),
        'communication_preference': np.random.choice(['email', 'phone', 'mail'], n_records),
        'last_contact_date': [datetime.now() - timedelta(days=np.random.randint(1, 100)) for _ in range(n_records)],
        'payment_probability': np.random.uniform(0, 1, n_records).round(4)
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv('C:/Users/Ramkishor/Downloads/New folder/CogniHack/debt_collection_data.csv', index=False)

    print("Data generated and saved to 'debt_collection_data.csv'")