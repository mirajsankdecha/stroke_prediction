# app/schema.py
from pydantic import BaseModel, Field, validator
from typing import Optional

class StrokePredictionRequest(BaseModel):
    id: Optional[int] = Field(None, description="Patient ID")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    age: float = Field(..., gt=0, lt=120, description="Age in years")
    hypertension: int = Field(..., ge=0, le=1, description="0: No hypertension, 1: Has hypertension")
    heart_disease: int = Field(..., ge=0, le=1, description="0: No heart disease, 1: Has heart disease")
    ever_married: str = Field(..., description="Yes or No")
    work_type: str = Field(..., description="Private, Self-employed, Govt_job, children, Never_worked")
    Residence_type: str = Field(..., description="Urban or Rural")
    avg_glucose_level: float = Field(..., gt=0, description="Average glucose level")
    bmi: Optional[float] = Field(None, gt=0, description="Body Mass Index")
    smoking_status: str = Field(..., description="formerly smoked, never smoked, smokes, Unknown")

    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female', 'Other']:
            raise ValueError('Gender must be Male, Female, or Other')
        return v

    @validator('ever_married')
    def validate_married(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('ever_married must be Yes or No')
        return v

    @validator('Residence_type')
    def validate_residence(cls, v):
        if v not in ['Urban', 'Rural']:
            raise ValueError('Residence_type must be Urban or Rural')
        return v

    @validator('work_type')
    def validate_work_type(cls, v):
        valid_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        if v not in valid_types:
            raise ValueError(f'work_type must be one of {valid_types}')
        return v

    @validator('smoking_status')
    def validate_smoking(cls, v):
        valid_statuses = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
        if v not in valid_statuses:
            raise ValueError(f'smoking_status must be one of {valid_statuses}')
        return v

class StrokePredictionResponse(BaseModel):
    stroke_prediction: int = Field(..., description="0: No stroke risk, 1: Stroke risk")
    stroke_probability: float = Field(..., description="Probability of stroke (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High risk")

class HealthStatus(BaseModel):
    status: str
    message: str
