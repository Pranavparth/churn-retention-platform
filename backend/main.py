from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI(title="Churn Retention API")

# Setup CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold artifacts
model = None
explainer = None
preprocessors = None
df_raw = None

@app.on_event("startup")
def load_artifacts():
    global model, explainer, preprocessors, df_raw
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, 'models', 'xgboost_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, 'models', 'shap_explainer.pkl'), 'rb') as f:
        explainer = pickle.load(f)
    with open(os.path.join(base_dir, 'models', 'preprocessors.pkl'), 'rb') as f:
        preprocessors = pickle.load(f)
        
    df_raw = pd.read_csv(os.path.join(base_dir, 'data', 'telecom_churn.csv'))
    
def preprocess_input(data_dict):
    """Encodes and scales raw input to feed into the model."""
    df_temp = pd.DataFrame([data_dict])
    
    for col in preprocessors['cat_cols']:
        encoder = preprocessors['encoders'][col]
        # Handle unseen labels by assigning the first class safely
        if df_temp[col][0] not in encoder.classes_:
             df_temp[col][0] = encoder.classes_[0]
        df_temp[col] = encoder.transform(df_temp[col])
        
    df_temp[preprocessors['num_cols']] = preprocessors['scaler'].transform(df_temp[preprocessors['num_cols']])
    return df_temp[preprocessors['feature_names']]

@app.get("/api/customers")
def get_customers(limit: int = 50):
    """Return a list of customers to display on the frontend."""
    # We will score them real quick so frontend has baseline churn risks
    sample = df_raw.head(limit).copy()
    
    customers = []
    for _, row in sample.iterrows():
        input_data = row.drop(['customer_id', 'churn']).to_dict()
        X_encoded = preprocess_input(input_data)
        prob = float(model.predict_proba(X_encoded)[0, 1])
        customers.append({
            "customer_id": row['customer_id'],
            "churn_risk": round(prob, 3),
            "monthly_charges": row['monthly_charges'],
            "tenure": row['tenure'],
            "support_calls": row['support_calls']
        })
        
    # Sort by risk descending
    customers.sort(key=lambda x: x["churn_risk"], reverse=True)
    return {"customers": customers}

@app.get("/api/customer/{customer_id}")
def get_customer_details(customer_id: str):
    user_row = df_raw[df_raw['customer_id'] == customer_id]
    if len(user_row) == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    user_dict = user_row.iloc[0].drop(['customer_id', 'churn']).to_dict()
    X_encoded = preprocess_input(user_dict)
    
    prob = float(model.predict_proba(X_encoded)[0, 1])
    shap_values = explainer(X_encoded)
    
    # Extract shap values for the positive class (churn)
    # XGBoost shap values from TreeExplainer might be raw log-odds
    # SHAP value dimension for binary classification depends on version, usually it's just local log odds.
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0] # or 1, tree depending
    
    contributions = []
    for i, feature in enumerate(preprocessors['feature_names']):
        val = shap_values.values[0][i]
        # If multinomial is returned, pick correctly. Assuming standard shap behavior.
        if isinstance(val, np.ndarray):
             val = val[-1] # take positive class
        contributions.append({
            "feature": feature,
            "value": round(float(val), 4),
            "raw_value": user_dict[feature]
        })
        
    return {
        "customer_id": customer_id,
        "features": user_dict,
        "churn_probability": round(prob, 3),
        "shap_base_value": round(float(base_value), 3) if not isinstance(base_value, list) else base_value,
        "shap_contributions": contributions
    }

class WhatIfRequest(BaseModel):
    customer_id: str
    features: dict

@app.post("/api/predict_what_if")
def predict_what_if(req: WhatIfRequest):
    """Recalculates probability and SHAP values given altered features."""
    X_encoded = preprocess_input(req.features)
    prob = float(model.predict_proba(X_encoded)[0, 1])
    shap_values = explainer(X_encoded)
    
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[-1]
    
    contributions = []
    for i, feature in enumerate(preprocessors['feature_names']):
        val = shap_values.values[0][i]
        if isinstance(val, np.ndarray):
             val = val[-1]
        contributions.append({
            "feature": feature,
            "value": round(float(val), 4),
            "raw_value": req.features[feature]
        })
        
    return {
        "churn_probability": round(prob, 3),
        "shap_contributions": contributions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
