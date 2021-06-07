# 1. uvicorn main:app --reload

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/text_analysis", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("text_analysis.html", {"request": request})


@app.get("/world_analysis", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("world_analysis.html", {"request": request})


@app.get("/user_analysis", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("user_analysis.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app)