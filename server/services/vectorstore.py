import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"   # must be lowercase + hyphen only

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

# Ensure index exists
existing_indexes = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # same as MiniLM
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

def upsert_vectors(vectors: list[tuple[str, list[float], dict]]):
    """Upsert list of (id, embedding, metadata) into Pinecone"""
    if vectors:
        index.upsert(vectors=vectors)

def query_vector(query_embedding: list[float], top_k: int = 5):
    """Query Pinecone with an embedding"""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results.matches
