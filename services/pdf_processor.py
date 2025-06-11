# import requests
# import PyPDF2
# import os

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

# import pdfplumber

# def extract_text_from_pdf(pdf_file):
#     """
#     Extract text from a PDF file using pdfplumber.
#     """
#     try:
#         text = ""
#         with pdfplumber.open(pdf_file) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#         return text
#     except Exception as e:
#         raise Exception(f"Error extracting text from PDF: {str(e)}")


# import pdfplumber
# import logging

# logging.basicConfig(level=logging.INFO)

# def extract_text_from_pdf(pdf_file):
#     try:
#         text = ""
#         with pdfplumber.open(pdf_file) as pdf:
#             for page in pdf.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
#         if not text.strip():
#             logging.warning("No text extracted from the PDF.")
#         return text
#     except Exception as e:
#         logging.error(f"Error extracting text from PDF: {str(e)}")
#         raise Exception(f"Error extracting text from PDF: {str(e)}")

import pdfplumber
import logging
import re

logging.basicConfig(level=logging.INFO)

def clean_text(text):
    # Fix common issues from OCR or malformed PDFs
    text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = text.replace('\u2022', '-')  # Bullet point fix
    text = re.sub(r'(?<=\w)- (?=\w)', '', text)  # Join hyphenated line breaks
    return text.strip()

def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            logging.warning("No text extracted from the PDF.")
        return clean_text(text)
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {str(e)}")
