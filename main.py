# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import routers
from app.controllers import router as api_router
from app.index_controllers import index_router as in_ro

import pickle
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Stroke Prediction API",
    description="API that predicts the probability of stroke based on patient health data.",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="frontend"), name="static")
templates = Jinja2Templates(directory="templates")

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        with open("model/best_stroke_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ Stroke Prediction Model loaded successfully!")
    except Exception as e:
        print("❌ Error loading the model:", e)

# Include routers
app.include_router(api_router)
app.include_router(in_ro)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
