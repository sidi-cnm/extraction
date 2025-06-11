from transformers import AutoTokenizer, AutoModel
import torch

# Load the BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding as a representation
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Example usage
sample_text = "COVID-19 is caused by the SARS-CoV-2 virus."
embedding = get_embedding(sample_text)
print(embedding)
