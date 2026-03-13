# 🔮 Interactive Customer Churn & Retention Platform

A Data Science application simulating a Telco company's customer retention dashboard. It's built with **Python/FastAPI** (Machine Learning/Backend) and **Next.js** (Frontend UI) featuring an ultra-custom, glassmorphic dark-mode interface built on pure Vanilla CSS.

![Next.js](https://img.shields.io/badge/Next.js-14-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-success)

## 🌟 What-If Simulator & Explainable AI
Unlike static machine learning notebooks, this project offers an interactive **What-If Simulator**.
- **Explainability**: Using `shap.TreeExplainer`, each user's profile is broken down to highlight exactly *why* they are a churn risk (e.g. "High Monthly Charges", "Month-to-month contract").
- **Live Interventions**: The user interface allows you directly manipulate a customer's `Monthly Charges`, `Support Calls`, or `Contract Type`. Changes stream to the FastAPI backend and instantly recalculate the predicted churn probability—demonstrating how business interventions can save high-risk accounts.

---

## 🏗️ Technical Architecture
### 1. Data Science & Backend (`/backend`)
- **Data Generation**: Uses a custom script (`data_generator.py`) to synthesize a realistic, mathematically correlated 5000-record dataset.
- **Modeling**: Trains a `RandomForestClassifier` to predict binary churn status.
- **Explainability Artifacts**: Fits and serializes a `SHAP` explainer alongside standard Scalers and LabelEncoders inside `/models`.
- **API**: A `FastAPI` application exposing `GET /api/customers` for the dashboard overview and `POST /api/predict_what_if` for real-time recalculations.

### 2. Premium Frontend (`/frontend`)
- **Framework**: `Next.js` Application Router.
- **Aesthetic**: Entirely custom styled without external UI libraries. Heavy use of CSS variables, `backdrop-filter: blur(12px)` for glassmorphism, and a sleek dark mode.
- **Components**: `ShapVisualization.tsx` renders dynamic, animated CSS bars to reflect the continuously calculating `SHAP` baseline values.

---

## 🚀 How to Run Locally
 
### 1. Start the Machine Learning API (Backend)
Navigate to the backend directory and set up the Python environment:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you wish to regenerate the data and re-train the models:
```bash
python data_generator.py
python train_model.py
```

Start the FastAPI server:
```bash
python main.py
```
*The server will start running on http://127.0.0.1:8000.*

### 2. Start the Premium Dashboard (Frontend)
Open a new terminal window and navigate to the frontend folder:
```bash
cd frontend
npm install
```
Start the Next.js development server:
```bash
npm run dev
```
*The dashboard is live! Open **http://localhost:3000** in your web browser to explore.*
