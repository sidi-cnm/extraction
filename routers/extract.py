# # routers/extract.py

# from fastapi import APIRouter, HTTPException
# from fastapi.responses import JSONResponse
# from services.biobert import extract_entities
# import requests
# import PyPDF2
# import os

# router = APIRouter()

# def download_and_extract_text(pdf_url):
#     try:
#         # Download the PDF file
#         response = requests.get(pdf_url)
#         if response.status_code != 200:
#             raise HTTPException(status_code=404, detail="PDF not found")
        
#         # Save the PDF locally
#         pdf_path = "temp.pdf"
#         with open(pdf_path, "wb") as f:
#             f.write(response.content)
        
#         # Extract text from the PDF
#         with open(pdf_path, "rb") as f:
#             reader = PyPDF2.PdfReader(f)
#             text = ""
#             for page in reader.pages:
#                 text += page.extract_text() or ""
        
#         # Clean up the temporary file
#         os.remove(pdf_path)
#         return text
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# @router.post("/extract/")
# async def extract_entities_from_pdf(pdf_url: str):
#     # Step 1: Download and extract text from the PDF
#     text = download_and_extract_text(pdf_url)
    
#     # Step 2: Use BioBERT to extract named entities
#     entities = extract_entities(text)
    
#     # Step 3: Return the extracted entities
#     return {"text": text, "entities": entities}


# from fastapi import APIRouter, HTTPException
# from services.biobert import extract_entities
# from services.pdf_processor import download_and_extract_text
# from schemas.request_models import ExtractRequest

# router = APIRouter()

# @router.post("/extract/")
# async def extract_entities_from_pdf(request: ExtractRequest):
#     pdf_url = request.pdf_url
    
#     # Step 1: Download and extract text from the PDF
#     text = download_and_extract_text(pdf_url)
    
#     # Step 2: Use BioBERT to extract named entities
#     entities = extract_entities(text)
    
#     # Step 3: Return the extracted entities
#     return {"text": text, "entities": entities}

#=> from computer 
# from fastapi import APIRouter, HTTPException, File, UploadFile
# from services.biobert import extract_entities
# from services.pdf_processor import extract_text_from_pdf
# import time

# router = APIRouter()

# @router.post("/extract/")
# async def extract_entities_from_pdf(file: UploadFile = File(...)):
#     # Start the timer
#     start_time = time.time()
    
#     try:
#         # Step 1: Extract text from the uploaded PDF file
#         text = extract_text_from_pdf(file.file)
        
#         # Step 2: Use BioBERT to extract named entities
#         entities = extract_entities(text)
        
#         # Calculate total execution time
#         execution_time = time.time() - start_time
        
#         # Return the extracted entities and execution time
#         return {
#             "text": text,
#             "entities": entities,
#             "execution_time_seconds": round(execution_time, 2)  # Rounded to 2 decimal places
#         }
#     except Exception as e:
#         # Log the error and execution time even if an exception occurs
#         execution_time = time.time() - start_time
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing request. Execution time: {round(execution_time, 2)} seconds. Error: {str(e)}"
#         )

#shunks:

# import time
# from fastapi import APIRouter, HTTPException, File, UploadFile

# from services.biobert import extract_entities
# from services.pdf_processor import extract_text_from_pdf
# router = APIRouter()
# @router.post("/extract/")
# async def extract_entities_from_pdf(file: UploadFile = File(...)):
#     # Start the timer
#     start_time = time.time()
    
#     try:
#         # Step 1: Extract text from the uploaded PDF file
#         text = extract_text_from_pdf(file.file)
        
#         # Step 2: Use BioBERT to extract named entities
#         entities = extract_entities(text)  # Handles chunking internally
        
#         # Calculate total execution time
#         execution_time = time.time() - start_time
        
#         # Return the extracted entities and execution time
#         return {
#             "text": text,
#             "entities": entities,
#             "execution_time_seconds": round(execution_time, 2)  # Rounded to 2 decimal places
#         }
#     except Exception as e:
#         # Log the error and execution time even if an exception occurs
#         execution_time = time.time() - start_time
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing request. Execution time: {round(execution_time, 2)} seconds. Error: {str(e)}"
#         )


# from fastapi import APIRouter, HTTPException, File, UploadFile
# from services.biobert import extract_entities
# from services.pdf_processor import extract_text_from_pdf
# import time

# router = APIRouter()

# @router.post("/extract/")
# async def extract_entities_from_pdf(file: UploadFile = File(...)):
#     # Start the timer
#     start_time = time.time()
    
#     try:
#         # Step 1: Extract text from the uploaded PDF file
#         text = extract_text_from_pdf(file.file)
        
#         # Validate that the extracted text is not empty
#         if not text.strip():
#             raise HTTPException(status_code=400, detail="The uploaded PDF contains no extractable text.")
        
#         # Step 2: Use BioBERT to extract named entities
#         entities = extract_entities(text)  # Handles chunking internally
        
#         # Calculate total execution time
#         execution_time = time.time() - start_time
        
#         # Return the extracted entities and execution time
#         return {
#             "text": text,
#             "entities": entities,
#             "execution_time_seconds": round(execution_time, 2)  # Rounded to 2 decimal places
#         }
#     except Exception as e:
#         # Log the error and execution time even if an exception occurs
#         execution_time = time.time() - start_time
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing request. Execution time: {round(execution_time, 2)} seconds. Error: {str(e)}"
#         )

# from fastapi import APIRouter, HTTPException, File, UploadFile
# from services.biobert import extract_entities_or_embeddings
# from services.pdf_processor import extract_text_from_pdf
# import time
# import logging

# logging.basicConfig(level=logging.INFO)
# router = APIRouter()

# @router.post("/extract/")
# async def extract_entities_or_embeddings_from_pdf(file: UploadFile = File(...)):
#     start_time = time.time()
#     try:
#         text = extract_text_from_pdf(file.file)
#         if not text.strip():
#             logging.error("Uploaded PDF contains no extractable text.")
#             raise HTTPException(status_code=400, detail="The uploaded PDF contains no extractable text.")
        
#         logging.info(f"Extracted text length: {len(text)} characters")
#         result = extract_entities_or_embeddings(text)

#         execution_time = time.time() - start_time
#         return {
#             "text": text,
#             "result": result,
#             "execution_time_seconds": round(execution_time, 2)
#         }

#     except Exception as e:
#         execution_time = time.time() - start_time
#         logging.error(f"Error processing request. Execution time: {execution_time:.2f} seconds. Error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing request. Execution time: {round(execution_time, 2)} seconds. Error: {str(e)}"
#         )

from fastapi import APIRouter, HTTPException, File, UploadFile
from services.biobert import extract_entities
from services.pdf_processor import extract_text_from_pdf
import time
import logging

logging.basicConfig(level=logging.INFO)
router = APIRouter()

@router.post("/extract/")
async def extract_entities_or_embeddings_from_pdf(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        text = extract_text_from_pdf(file.file)

        if not text.strip():
            logging.error("Uploaded PDF contains no extractable text.")
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no extractable text.")
        
        logging.info(f"Extracted text length: {len(text)} characters")

        result = extract_entities(text)

        execution_time = time.time() - start_time
        return {
            "text": text,
            "result": result,
            "execution_time_seconds": round(execution_time, 2)
        }

    except Exception as e:
        execution_time = time.time() - start_time
        logging.error(f"Error processing request. Execution time: {execution_time:.2f} seconds. Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request. Execution time: {round(execution_time, 2)} seconds. Error: {str(e)}"
        )
