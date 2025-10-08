# app/index_controllers.py
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

index_router = APIRouter(tags=["Frontend"])
templates = Jinja2Templates(directory="templates")

@index_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main prediction form"""
    return templates.TemplateResponse("index.html", {"request": request})

@index_router.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API documentation"""
    return {"message": "Visit /docs for API documentation"}
