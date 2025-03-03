# main.py
from fastapi import FastAPI

app = FastAPI(
    title="Extraction Service",
    description="Service pour extraire des embeddings à partir de dossiers medicaux",
    version="0.1.0"
)

@app.get("/")
def read_root():
    return {"message": "Hello from Extraction Service!"}
