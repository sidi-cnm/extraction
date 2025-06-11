# main.py

from fastapi import FastAPI
from routers.extract import router as extract_router

app = FastAPI(
    title="Extraction Service",
    description="Service pour extraire des embeddings à partir de dossiers médicaux",
    version="0.1.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello from Extraction Service!"}

# Include the extract router
app.include_router(extract_router, prefix="/api/v1")