# 1. uvicorn main:app --reload

from joblib import load
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

vectorizer = load("vectorizer.joblib")
model = load("stack_model.joblib")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/text_analysis", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("text_analysis.html", {"request": request})


@app.get("/user_analysis", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("user_analysis.html", {"request": request})


@app.get("/results_text_analysis", response_class=HTMLResponse)
async def index(request: Request):
    text = "I am extremely happy"
    prediction = model.predict(vectorizer.transform([text]))

    return templates.TemplateResponse("text_results.html", {"request": request, "prediction": prediction})


@app.get("/results_user_analysis", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("user_results.html", {"request": request})
