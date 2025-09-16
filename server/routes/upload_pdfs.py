import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from services.embeddings import get_embeddings
from services.vectorstore import upsert_vectors

router = APIRouter()

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload_pdfs/")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    file_paths = []

    # Save PDFs locally
    for file in files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(await file.read())
        file_paths.append(str(save_path))

    # Process each PDF
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        embeddings = get_embeddings(texts)
        vectors = [(id_, emb, {**meta, "text": text})
                   for id_, emb, meta, text in zip(ids, embeddings, metadatas, texts)]

        upsert_vectors(vectors)

    return {"message": "PDFs uploaded and indexed successfully"}
