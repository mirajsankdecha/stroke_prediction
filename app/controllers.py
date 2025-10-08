# app/controllers.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
from typing import Optional
import logging
import os

# Initialize router
api_router = APIRouter(prefix="/api", tags=["Stroke Prediction"])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load the ML model with multiple fallback paths for Render deployment"""
    model_paths = [
        # Production paths (for Render)
        "model/best_stroke_model.pkl",
        "./model/best_stroke_model.pkl",
        "best_stroke_model.pkl",
        # Local development path
        "C:/Users/Lenovo/Downloads/stroke_prediction/model/best_stroke_model.pkl",
        # Alternative paths
        "/opt/render/project/src/model/best_stroke_model.pkl",
        "./stroke_prediction/model/best_stroke_model.pkl"
    ]
    
    for path in model_paths:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    model = pickle.load(f)
                logger.info(f"✅ Model loaded successfully from: {path}")
                return model
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model from {path}: {e}")
            continue
    
    logger.error("❌ Failed to load model from any path")
    return None

# Load model globally
model = load_model()

# Input schema matching your frontend form
class StrokeInput(BaseModel):
    gender: str = Field(..., description="Male, Female, or Other")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    hypertension: int = Field(..., ge=0, le=1, description="0 or 1")
    heart_disease: int = Field(..., ge=0, le=1, description="0 or 1")
    ever_married: str = Field(..., description="Yes or No")
    work_type: str = Field(..., description="Work type category")
    Residence_type: str = Field(..., description="Urban or Rural")
    avg_glucose_level: float = Field(..., gt=0, description="Glucose level")
    bmi: Optional[float] = Field(None, gt=0, description="Body Mass Index")
    smoking_status: str = Field(..., description="Smoking status")

# Response schema
class StrokePredictionResponse(BaseModel):
    stroke_prediction: int = Field(..., description="0: Low risk, 1: High risk")
    stroke_probability: float = Field(..., description="Probability of stroke (0-1)")
    confidence: float = Field(..., description="Model confidence percentage")
    risk_level: str = Field(..., description="Low, Medium, or High")
    message: str = Field(..., description="Human readable message")

def encode_categorical_data(data: StrokeInput):
    """
    Convert categorical string data to numerical format for model prediction
    Returns exactly 11 features as expected by the model
    """
    try:
        # Gender encoding (Male=1, Female=0, Other=2)
        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        gender_encoded = gender_map.get(data.gender, 1)
        
        # Ever married encoding (Yes=1, No=0)
        married_map = {"Yes": 1, "No": 0}
        married_encoded = married_map.get(data.ever_married, 0)
        
        # Work type encoding
        work_map = {
            "Private": 0,
            "Self-employed": 1, 
            "Govt_job": 2,
            "children": 3,
            "Never_worked": 4
        }
        work_encoded = work_map.get(data.work_type, 0)
        
        # Residence type encoding (Urban=1, Rural=0)
        residence_map = {"Urban": 1, "Rural": 0}
        residence_encoded = residence_map.get(data.Residence_type, 1)
        
        # Smoking status encoding
        smoking_map = {
            "never smoked": 0,
            "formerly smoked": 1,
            "smokes": 2,
            "Unknown": 3
        }
        smoking_encoded = smoking_map.get(data.smoking_status, 0)
        
        # Handle missing BMI - use dataset median
        bmi_value = data.bmi if data.bmi is not None else 28.1
        
        # BMI category as 11th feature (0=underweight, 1=normal, 2=overweight, 3=obese)
        if bmi_value < 18.5:
            bmi_category = 0  # underweight
        elif bmi_value < 25:
            bmi_category = 1  # normal
        elif bmi_value < 30:
            bmi_category = 2  # overweight
        else:
            bmi_category = 3  # obese
        
        # Return exactly 11 features
        features = [
            gender_encoded,         # 1. gender
            data.age,              # 2. age  
            data.hypertension,     # 3. hypertension
            data.heart_disease,    # 4. heart_disease
            married_encoded,       # 5. ever_married
            work_encoded,          # 6. work_type
            residence_encoded,     # 7. Residence_type
            data.avg_glucose_level, # 8. avg_glucose_level
            bmi_value,             # 9. bmi
            smoking_encoded,       # 10. smoking_status
            bmi_category           # 11. bmi_category
        ]
        
        logger.info(f"🔢 Encoded features (count: {len(features)}): {features}")
        return features
        
    except Exception as e:
        logger.error(f"❌ Error encoding data: {e}")
        raise HTTPException(status_code=400, detail=f"Data encoding error: {str(e)}")

@api_router.post("/predict", response_model=StrokePredictionResponse)
def predict_stroke(data: StrokeInput):
    """Predict stroke probability based on patient data"""
    try:
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="ML Model not available. Please check model file exists."
            )
        
        logger.info(f"📥 Received prediction request: {data.dict()}")
        
        # Encode categorical data to numerical format
        features_list = encode_categorical_data(data)
        
        # Convert to numpy array with correct shape for model
        features = np.array([features_list])
        
        logger.info(f"📊 Model input shape: {features.shape}")
        logger.info(f"🎯 Model input: {features}")
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities
        try:
            probability = model.predict_proba(features)[0]
            stroke_probability = float(probability[1]) if len(probability) > 1 else float(probability[0])
            confidence_percentage = max(probability) * 100
            logger.info(f"📈 Prediction probabilities: {probability}")
        except Exception as prob_error:
            logger.warning(f"⚠️ predict_proba failed: {prob_error}")
            # Fallback for models without predict_proba
            stroke_probability = float(prediction)
            confidence_percentage = 85.0 if prediction == 1 else 92.0
        
        # Determine risk level
        if stroke_probability >= 0.7:
            risk_level = "High"
        elif stroke_probability >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Create human readable message
        if prediction == 1:
            message = f"High stroke risk detected ({stroke_probability*100:.1f}% probability)"
        else:
            message = f"Low stroke risk ({stroke_probability*100:.1f}% probability)"
        
        response = StrokePredictionResponse(
            stroke_prediction=int(prediction),
            stroke_probability=stroke_probability,
            confidence=float(confidence_percentage),
            risk_level=risk_level,
            message=message
        )
        
        logger.info(f"✅ Prediction response: {response.dict()}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.get("/health")
def api_health_check():
    """API Health check endpoint for monitoring"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "api_status": "healthy",
        "model_status": model_status,
        "model_loaded": model is not None,
        "expected_features": 11,
        "feature_names": [
            "gender", "age", "hypertension", "heart_disease", 
            "ever_married", "work_type", "Residence_type", 
            "avg_glucose_level", "bmi", "smoking_status", "bmi_category"
        ],
        "message": f"Stroke Prediction API is running on Render - Model: {model_status}",
        "environment": "production"
    }

@api_router.get("/model-info")
def get_model_info():
    """Get detailed model information"""
    try:
        model_type = type(model).__name__ if model else "Not loaded"
        return {
            "model_type": model_type,
            "model_loaded": model is not None,
            "input_features": 11,
            "categorical_encodings": {
                "gender": {"Male": 1, "Female": 0, "Other": 2},
                "ever_married": {"Yes": 1, "No": 0},
                "work_type": {
                    "Private": 0, "Self-employed": 1, "Govt_job": 2,
                    "children": 3, "Never_worked": 4
                },
                "Residence_type": {"Urban": 1, "Rural": 0},
                "smoking_status": {
                    "never smoked": 0, "formerly smoked": 1, 
                    "smokes": 2, "Unknown": 3
                },
                "bmi_category": {
                    "underweight": 0, "normal": 1, 
                    "overweight": 2, "obese": 3
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
