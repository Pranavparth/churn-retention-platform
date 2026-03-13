import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap
import pickle
import os

def prepare_data(df):
    # Categorical columns to encode
    cat_cols = ['contract_type', 'internet_service', 'payment_method']
    
    encoders = {}
    df_encoded = df.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Scale numerical columns
    num_cols = ['tenure', 'monthly_charges', 'total_charges', 'support_calls']
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df_encoded, encoders, scaler, cat_cols, num_cols

def train_and_save_model():
    print("Loading data...")
    df = pd.read_csv('data/telecom_churn.csv')
    
    # We don't train on customer_id
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    
    print("Preparing features...")
    df_encoded, encoders, scaler, cat_cols, num_cols = prepare_data(df)
    
    X_encoded = df_encoded.drop(columns=['customer_id', 'churn'])
    feature_names = X_encoded.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Save artifacts
    print("Saving model and artifacts...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
        
    with open('models/preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'encoders': encoders,
            'scaler': scaler,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'feature_names': feature_names
        }, f)
        
    print("All artifacts saved successfully to models/")

if __name__ == "__main__":
    train_and_save_model()
