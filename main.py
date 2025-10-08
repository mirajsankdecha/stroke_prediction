# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Import routers
from app.controllers import api_router
from app.index_controllers import index_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Stroke Prediction API",
    description="Advanced Machine Learning API for stroke risk prediction based on patient health data.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (with error handling)
try:
    if os.path.exists("frontend"):
        app.mount("/static", StaticFiles(directory="frontend"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(api_router)
app.include_router(index_router)

# Global health check (separate from API health check)
@app.get("/health")
async def root_health_check():
    """Root level health check for Render monitoring"""
    return {
        "status": "healthy",
        "message": "AI Stroke Prediction API is running on Render",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Use environment PORT for Render, fallback to 8000 for local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",  # Important: Use 0.0.0.0 for Render
        port=port, 
        reload=False,  # Disable reload in production
        log_level="info"
    )
