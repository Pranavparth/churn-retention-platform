import pandas as pd
import numpy as np
import os

def generate_telecom_data(num_samples=5000):
    np.random.seed(42)
    
    # Generate features
    tenure = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=num_samples)
    
    # Intentionally add correlation: lower tenure & higher charges -> more support calls
    base_calls = np.random.poisson(lam=1.5, size=num_samples)
    high_charge_penalty = np.where(monthly_charges > 80, 1, 0)
    low_tenure_penalty = np.where(tenure < 12, 1, 0)
    support_calls = base_calls + high_charge_penalty + low_tenure_penalty
    support_calls = np.clip(support_calls, 0, 10)
    
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=num_samples)
    total_charges = np.where(total_charges < 0, 0, total_charges)
    
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples, p=[0.5, 0.25, 0.25])
    internet_service = np.random.choice(['Fiber optic', 'DSL', 'No'], size=num_samples, p=[0.45, 0.35, 0.20])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], size=num_samples, p=[0.35, 0.2, 0.25, 0.2])
    
    # Generate target (churn) based on a hidden logic
    churn_prob = np.zeros(num_samples)
    
    # Higher charges -> higher churn
    churn_prob += (monthly_charges - 20) / 100 * 0.3
    
    # Longer tenure -> lower churn
    churn_prob -= (tenure / 72) * 0.4
    
    # High support calls -> much higher churn
    churn_prob += (support_calls / 10) * 0.5
    
    # Month-to-month contract -> higher churn
    churn_prob += np.where(contract_type == 'Month-to-month', 0.25, 0)
    
    # Fiber optic sometimes has more churn in realistic datasets due to price/competition
    churn_prob += np.where(internet_service == 'Fiber optic', 0.1, 0)
    
    # Electronic check historically higher churn
    churn_prob += np.where(payment_method == 'Electronic check', 0.1, 0)
    
    # Normalize and create binary target
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'customer_id': [f'CUST-{i:04d}' for i in range(num_samples)],
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'support_calls': support_calls,
        'contract_type': contract_type,
        'internet_service': internet_service,
        'payment_method': payment_method,
        'churn': churn
    })
    
    return df

if __name__ == "__main__":
    df = generate_telecom_data()
    # Save to data directory
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/telecom_churn.csv', index=False)
    print(f"Generated {len(df)} records and saved to data/telecom_churn.csv")
    print("\nSample churn rate:", df['churn'].mean())
