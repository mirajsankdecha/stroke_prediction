# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import routers - FIXED IMPORT NAMES
from app.controllers import api_router
from app.index_controllers import index_router

import pickle
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Stroke Prediction API",
    description="API that predicts the probability of stroke based on patient health data.",
    version="1.0.0"
)

# Mount static files and templates
try:
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
except:
    pass  # In case frontend directory doesn't exist yet

templates = Jinja2Templates(directory="templates")

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        with open("C:/Users/Lenovo/Downloads/stroke_prediction/model/best_stroke_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ Stroke Prediction Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading the model: {e}")
        print("🔧 Train the model first using: python train_model.py")

# Include routers - FIXED NAMES
app.include_router(api_router)
app.include_router(index_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
