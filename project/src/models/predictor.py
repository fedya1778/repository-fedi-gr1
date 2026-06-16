import logging
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

logger = logging.getLogger("api_service.predictor")

class ModelPredictor:
    def __init__(self, model_path='artifacts/models/production_model.pkl'):
        base_dir = Path(__file__).resolve().parent.parent.parent
        full_path = base_dir / model_path
        
        logger.info(f"Loading model from {full_path}")
        if not full_path.exists():
            logger.critical(f"Model file not found at: {full_path}")
            raise FileNotFoundError(f"Файл модели не найден")
            
        self.model = joblib.load(full_path)
        logger.info("Model loaded successfully")

    def predict(self, client_data: dict):
        age = client_data.get("Customer_Age", 0)
        income = client_data.get("Annual_Income", 0)
        credit_score_raw = client_data.get("Credit_Score", 0)

        if age < 18 or age > 100 or income <= 0 or credit_score_raw <= 0:
            logger.warning(
                f"Request rejected by Policy Rules. Age: {age}, Income: {income}, Score: {credit_score_raw}"
            )
            return {
                "credit_score": 300,
                "probability_of_default": 1.0,
                "status": "High Risk (Rejected by Policy Rules)"
            }

        df = pd.DataFrame([client_data])
        
        categorical_cols = ['Gender', 'Purchase_Category', 'BNPL_Provider', 
                            'Device_Type', 'Connection_Type', 'Browser']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        
        prob_default = float(self.model.predict_proba(df)[0][1])
        prob_default = np.clip(prob_default, 0.0001, 0.9999)
        
        A = 712.86
        B = 28.85
        score = A - B * np.log(prob_default / (1 - prob_default))
        score = int(np.clip(score, 300, 850))
        
        if prob_default > 0.30:
            status = "High Risk"
        elif 0.10 <= prob_default <= 0.30:
            status = "Medium Risk"
        else:
            status = "Low Risk"
            
        return {
            "credit_score": score,
            "probability_of_default": round(prob_default, 4),
            "status": status
        }