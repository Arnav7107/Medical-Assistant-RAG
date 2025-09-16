# import os
# from langchain_huggingface import HuggingFaceEndpointEmbeddings

# # HuggingFace Embedding model
# # Make sure you set HUGGINGFACE_API_TOKEN in .env
# embed_model = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2"
# )

# def get_embedding(text: str):
#     """Generate embedding for a single text input"""
#     return embed_model.embed_documents([text])[0]

# def get_embeddings(texts: list[str]):
#     """Generate embeddings for multiple texts"""
#     return embed_model.embed_documents(texts)


import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Hugging Face token from .env
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ Missing Hugging Face token. Please set HUGGINGFACEHUB_API_TOKEN in your .env")

# HuggingFace Inference API Embedding model
embed_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

def get_embedding(text: str):
    """Generate embedding for a single text input"""
    return embed_model.embed_documents([text])[0]

def get_embeddings(texts: list[str]):
    """Generate embeddings for multiple texts"""
    return embed_model.embed_documents(texts)
