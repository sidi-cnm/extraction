# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# # Lazy-load the model and tokenizer
# _tokenizer = None
# _model = None
# _ner_pipeline = None

# def get_ner_pipeline():
#     global _tokenizer, _model, _ner_pipeline
#     if _ner_pipeline is None:
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         _model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
#         _ner_pipeline = pipeline(
#             "ner",
#             model=_model,
#             tokenizer=_tokenizer,
#             aggregation_strategy="simple"
#         )
#     return _ner_pipeline

# def extract_entities(text):
#     """
#     Extract entities from text using BioBERT NER pipeline.
#     """
#     ner_pipeline = get_ner_pipeline()
#     return ner_pipeline(text)


# => this code ysal7 mouchkilt tokens , bert wll ey model transformer yegba8 la 7d mou7aded (512 -v7altne )
# so lehi n3adlu chunks 
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# # Lazy-load the model and tokenizer
# _tokenizer = None
# _model = None
# _ner_pipeline = None

# def get_ner_pipeline():
#     global _tokenizer, _model, _ner_pipeline
#     if _ner_pipeline is None:
#         print("Loading BioBERT model...")
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         _model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
#         _ner_pipeline = pipeline(
#             "ner",
#             model=_model,
#             tokenizer=_tokenizer,
#             aggregation_strategy="simple"
#         )
#     return _ner_pipeline

# def extract_entities(text, max_chunk_size=512):
#     """
#     Extract entities from text using BioBERT NER pipeline.
#     Splits text into chunks to avoid exceeding the model's max sequence length.
#     """
#     ner_pipeline = get_ner_pipeline()
    
#     # Tokenize the text to determine its length
#     tokens = _tokenizer.tokenize(text)
#     if len(tokens) <= max_chunk_size:
#         # If the text fits within the max length, process it directly
#         return ner_pipeline(text)
    
#     # Otherwise, split the text into chunks
#     chunks = []
#     current_chunk = ""
#     for token in tokens:
#         if len(_tokenizer.tokenize(current_chunk)) + len(_tokenizer.tokenize(token)) <= max_chunk_size:
#             current_chunk += token + " "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = token + " "
#     if current_chunk:
#         chunks.append(current_chunk.strip())
    
#     # Process each chunk and combine the results
#     all_entities = []
#     for chunk in chunks:
#         entities = ner_pipeline(chunk)
#         all_entities.extend(entities)
    
#     return all_entities

# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# # Lazy-load the model and tokenizer
# _tokenizer = None
# _model = None
# _ner_pipeline = None

# def get_ner_pipeline():
#     global _tokenizer, _model, _ner_pipeline
#     if _ner_pipeline is None:
#         print("Loading BioBERT model...")
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         _model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
#         _ner_pipeline = pipeline(
#             "ner",
#             model=_model,
#             tokenizer=_tokenizer,
#             aggregation_strategy="simple"
#         )
#     return _ner_pipeline

# def extract_entities(text, max_chunk_size=512):
#     """
#     Extract entities from text using BioBERT NER pipeline.
#     Splits text into chunks to avoid exceeding the model's max sequence length.
#     """
#     ner_pipeline = get_ner_pipeline()
    
#     # Tokenize the text to determine its length
#     tokens = _tokenizer.encode(text, add_special_tokens=False)  # Exclude special tokens for now
    
#     # Ensure the chunk size accounts for special tokens ([CLS], [SEP])
#     max_chunk_size -= 2  # Reserve space for [CLS] and [SEP]
    
#     # Split the text into chunks
#     chunks = []
#     current_chunk = []
#     for token_id in tokens:
#         if len(current_chunk) < max_chunk_size:
#             current_chunk.append(token_id)
#         else:
#             # Add the current chunk and start a new one
#             chunks.append(_tokenizer.decode(current_chunk))
#             current_chunk = [token_id]
#     if current_chunk:
#         chunks.append(_tokenizer.decode(current_chunk))
    
#     # Process each chunk and combine the results
#     all_entities = []
#     for chunk in chunks:
#         entities = ner_pipeline(chunk)
#         all_entities.extend(entities)
    
#     return all_entities


# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# # Lazy-load the model and tokenizer
# _tokenizer = None
# _model = None
# _ner_pipeline = None

# def get_ner_pipeline():
#     global _tokenizer, _model, _ner_pipeline
#     if _ner_pipeline is None:
#         print("Loading BioBERT model...")
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         _model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
#         _ner_pipeline = pipeline(
#             "ner",
#             model=_model,
#             tokenizer=_tokenizer,
#             aggregation_strategy="simple"
#         )
#     return _ner_pipeline

# def extract_entities(text, max_chunk_size=512):
#     """
#     Extract entities from text using BioBERT NER pipeline.
#     Splits text into chunks to avoid exceeding the model's max sequence length.
#     """
#     ner_pipeline = get_ner_pipeline()
    
#     # Tokenize the text to determine its length
#     tokens = _tokenizer.encode(text, add_special_tokens=False)  # Exclude special tokens for now
    
#     # Ensure the chunk size accounts for special tokens ([CLS], [SEP])
#     max_chunk_size -= 2  # Reserve space for [CLS] and [SEP]
    
#     # Split the text into chunks
#     chunks = []
#     current_chunk = []
#     for token_id in tokens:
#         if len(current_chunk) < max_chunk_size:
#             current_chunk.append(token_id)
#         else:
#             # Add the current chunk and start a new one
#             chunks.append(_tokenizer.decode(current_chunk))
#             current_chunk = [token_id]
#     if current_chunk:
#         chunks.append(_tokenizer.decode(current_chunk))
    
#     # Process each chunk and combine the results
#     all_entities = []
#     for chunk in chunks:
#         entities = ner_pipeline(chunk)
#         all_entities.extend(entities)
    
#     return all_entities


# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# # Lazy-load the model and tokenizer
# _tokenizer = None
# _model = None
# _ner_pipeline = None

# def get_ner_pipeline():
#     global _tokenizer, _model, _ner_pipeline
#     if _ner_pipeline is None:
#         print("Loading BioBERT model...")
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         _model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
#         _ner_pipeline = pipeline(
#             "ner",
#             model=_model,
#             tokenizer=_tokenizer,
#             aggregation_strategy="simple"
#         )
#     return _ner_pipeline

# def extract_entities(text, max_chunk_size=512):
#     """
#     Extract entities from text using BioBERT NER pipeline.
#     Splits text into chunks to avoid exceeding the model's max sequence length.
#     """
#     ner_pipeline = get_ner_pipeline()
    
#     # Validate that the input text is not empty
#     if not text.strip():
#         raise ValueError("Input text is empty.")
    
#     # Tokenize the text to determine its length
#     tokens = _tokenizer.encode(text, add_special_tokens=False)  # Exclude special tokens for now
    
#     # Split the text into chunks
#     chunks = []
#     current_chunk = []
#     for token_id in tokens:
#         if len(current_chunk) < max_chunk_size - 2:  # Reserve space for [CLS] and [SEP]
#             current_chunk.append(token_id)
#         else:
#             # Add the current chunk and start a new one
#             chunks.append(_tokenizer.decode(current_chunk))
#             current_chunk = [token_id]
#     if current_chunk:
#         chunks.append(_tokenizer.decode(current_chunk))
    
#     # Process each chunk and combine the results
#     all_entities = []
#     for chunk in chunks:
#         # Skip empty chunks
#         if not chunk.strip():
#             continue
        
#         # Use padding, truncation, and max_length to ensure compatibility with the model
#         inputs = _tokenizer(
#             chunk,
#             padding=True,
#             truncation=True,
#             max_length=max_chunk_size,
#             add_special_tokens=True,
#             return_tensors="pt"  # Return PyTorch tensors
#         )
        
#         # Pass the inputs through the NER pipeline
#         entities = ner_pipeline(inputs)
#         all_entities.extend(entities)
    
#     return all_entities


# from transformers import AutoTokenizer, AutoModel, pipeline
# import torch
# import logging

# logging.basicConfig(level=logging.INFO)

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
# _tokenizer = None
# _model = None
# _ner_pipeline = None

# def get_ner_pipeline():
#     global _tokenizer, _model, _ner_pipeline
#     if _ner_pipeline is None:
#         logging.info("Loading BioBERT model...")

#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         logging.info(f"Using device: {device}")  # 👈 Tu mets ça ici

#         _model = AutoModel.from_pretrained(MODEL_NAME).to(device)

#         _ner_pipeline = pipeline(
#             "ner",
#             model=_model,
#             tokenizer=_tokenizer,
#             aggregation_strategy="simple",
#             device=0 if torch.cuda.is_available() else -1
#         )
#     return _ner_pipeline

# def extract_entities(text, max_chunk_size=510):  # not 512
#     ner_pipeline = get_ner_pipeline()

#     tokens = _tokenizer.encode(text, add_special_tokens=False)

#     chunks = []
#     current_chunk = []
#     for token_id in tokens:
#         if len(current_chunk) < max_chunk_size:
#             current_chunk.append(token_id)
#         else:
#             chunks.append(current_chunk)
#             current_chunk = [token_id]
#     if current_chunk:
#         chunks.append(current_chunk)

#     all_entities = []
#     for token_ids in chunks:
#         try:
#             input_text = _tokenizer.decode(token_ids, skip_special_tokens=True)
#             if not input_text.strip():
#                 continue
#             entities = ner_pipeline(input_text)
#             all_entities.extend(entities)
#         except Exception as e:
#             logging.error(f"NER pipeline failed on chunk: '{input_text[:50]}...' with error: {str(e)}")
    
#     return all_entities

# def get_embedding(text):
#     if not text.strip():
#         raise ValueError("Input text is empty for embedding generation.")

#     inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
#         torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     )
#     with torch.no_grad():
#         outputs = _model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# def extract_entities_or_embeddings(text):
#     # Optionally, switch logic here if you want to support both embeddings and NER
#     return extract_entities(text)


# #sgill
# from transformers import AutoTokenizer, AutoModel, pipeline
# import torch
# import logging
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# # Lazy-load model and tokenizer
# _tokenizer = None
# _model = None
# _ner_pipeline = None
# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.info(f"Using device: {_device}")

# # def get_ner_pipeline():
# #     global _tokenizer, _model, _ner_pipeline
# #     if _ner_pipeline is None:
# #         logging.info("Loading BioBERT model...")
# #         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# #         _model = AutoModel.from_pretrained(MODEL_NAME).to(_device)
# #         _ner_pipeline = pipeline(
# #             "ner",
# #             model=_model,
# #             tokenizer=_tokenizer,
# #             aggregation_strategy="simple",
# #             device=0 if torch.cuda.is_available() else -1
# #         )
# #     return _ner_pipeline

# from transformers import pipeline

# # Set up the NER pipeline correctly
# ner_pipeline = pipeline(
#     "ner",
#     model="dmis-lab/biobert-base-cased-v1.1",
#     tokenizer="dmis-lab/biobert-base-cased-v1.1",
#     aggregation_strategy="simple",
#     truncation=True,
#     max_length=512,
#     device=0 if torch.cuda.is_available() else -1
# )

# def is_valid_chunk(text):
#     """Heuristic: skip chunks with very few alphabetic characters (likely numeric noise)."""
#     alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / (len(text) + 1e-6)
#     return alpha_ratio > 0.3

# def extract_entities(text, max_chunk_size=510):
#     """
#     Extract entities from text using BioBERT NER pipeline.
#     Splits text into chunks <= max_chunk_size tokens.
#     """
#     ner_pipeline = get_ner_pipeline()
#     tokens = _tokenizer.encode(text, add_special_tokens=False)
    
#     chunks = []
#     current_chunk = []
#     for token_id in tokens:
#         if len(current_chunk) < max_chunk_size:
#             current_chunk.append(token_id)
#         else:
#             chunks.append(current_chunk)
#             current_chunk = [token_id]
#     if current_chunk:
#         chunks.append(current_chunk)

#     all_entities = []
#     for token_ids in chunks:
#         input_text = _tokenizer.decode(token_ids, skip_special_tokens=True)
#         if not input_text.strip() or not is_valid_chunk(input_text):
#             continue
#         try:
#             entities = ner_pipeline(input_text)
#             all_entities.extend(entities)
#         except Exception as e:
#             logging.error(f"NER pipeline failed on chunk: '{input_text[:50]}...' with error: {str(e)}")
#     return all_entities

# def get_embedding(text):
#     """
#     Generate embeddings for input text using BioBERT.
#     Returns the [CLS] token embedding.
#     """
#     inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(_device)
#     with torch.no_grad():
#         outputs = _model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# def extract_entities_or_embeddings(text, mode="entities"):
#     if mode == "entities":
#         return extract_entities(text)
#     elif mode == "embedding":
#         return get_embedding(text)
#     else:
#         raise ValueError("Mode must be either 'entities' or 'embedding'")
# import logging
# import pdfplumber
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import torch
# import os
# import numpy as np

# # ========== Logging ==========
# logging.basicConfig(level=logging.INFO)

# # ========== Preprocessing PDF ==========
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
#         logging.info(f"Extracted text length: {len(text)} characters")
#         return text
#     except Exception as e:
#         logging.error(f"Error extracting text from PDF: {str(e)}")
#         raise Exception(f"Error extracting text from PDF: {str(e)}")

# # ========== NER Model ==========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.info(f"Device set to use {device}")

# model_name = "dmis-lab/biobert-base-cased-v1.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)  # adjust labels if needed
# model.to(device)

# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

# def chunk_text(text, chunk_size=512):
#     words = text.split()
#     return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# def extract_entities(text):
#     chunks = chunk_text(text)
#     all_entities = []

#     for chunk in chunks:
#         try:
#             if not chunk.strip():
#                 continue
#             entities = ner_pipeline(chunk, truncation=True)
#             for ent in entities:
#                 ent["score"] = float(ent["score"])
#             all_entities.extend(entities)
#         except Exception as e:
#             logging.error(f"NER pipeline failed on chunk: '{chunk[:100]}...' with error: {str(e)}")
#             continue

#     return all_entities

# # ========== FastAPI App ==========
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/api/v1/extract/")
# async def extract_entities_from_pdf(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         temp_pdf_path = "temp_upload.pdf"
#         with open(temp_pdf_path, "wb") as f:
#             f.write(contents)

#         text = extract_text_from_pdf(temp_pdf_path)
#         result = extract_entities(text)

#         os.remove(temp_pdf_path)

#         return {
#             "text": text,
#             "result": result,
#         }

#     except Exception as e:
#         logging.error(f"Failed to process PDF: {str(e)}")
#         return {"error": str(e)}


import logging
import pdfplumber
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import os
import numpy as np

# ========== Logging ==========
logging.basicConfig(level=logging.INFO)

# ========== Preprocessing PDF ==========
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
        logging.info(f"Extracted text length: {len(text)} characters")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Error extracting text from PDF: {str(e)}")

# ========== NER Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device set to use {device}")

model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)  # adjust labels if needed
model.to(device)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_entities(text):
    chunks = chunk_text(text)
    all_entities = []

    for chunk in chunks:
        try:
            if not chunk.strip():
                continue
            entities = ner_pipeline(chunk, truncation=True)
            for ent in entities:
                ent["score"] = float(ent["score"])
            all_entities.extend(entities)
        except Exception as e:
            logging.error(f"NER pipeline failed on chunk: '{chunk[:100]}...' with error: {str(e)}")
            continue

    return all_entities

# ========== FastAPI App ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/extract/")
async def extract_entities_from_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_pdf_path = "temp_upload.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(contents)

        text = extract_text_from_pdf(temp_pdf_path)
        result = extract_entities(text)

        os.remove(temp_pdf_path)

        return {
            "text": text,
            "result": result,
        }

    except Exception as e:
        logging.error(f"Failed to process PDF: {str(e)}")
        return {"error": str(e)}
