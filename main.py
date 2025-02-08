from fastapi import FastAPI
from routers import ocr

app = FastAPI()

app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])