# app/services.py
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class StrokeModelService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            model_path = Path("C:/Users/Lenovo/Downloads/stroke_prediction/model/best_stroke_model.pkl")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict_stroke(self, patient_data: dict):
        try:
            # Create DataFrame from input
            df = pd.DataFrame([patient_data])
            
            # Handle missing BMI
            if pd.isna(df['bmi'].iloc[0]) or df['bmi'].iloc[0] is None:
                df['bmi'] = df['bmi'].fillna(28.9)  # median BMI
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0][1]  # Probability of stroke
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return {
                "stroke_prediction": int(prediction),
                "stroke_probability": float(probability),
                "risk_level": risk_level
            }
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

# Global service instance
stroke_service = StrokeModelService()
