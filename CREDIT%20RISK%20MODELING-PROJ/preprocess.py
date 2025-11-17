import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.median())
    
    # Select features
    X = df[['age', 'income', 'loan_amount', 'credit_score',
            'employment_years', 'previous_defaults', 'debt_to_income_ratio']]
    
    y = df['loan_status']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    data = load_dataset("credit_risk_dataset.csv")
    X, y, scaler = preprocess_data(data)
    print("Preprocessing complete! Shape:", X.shape)
