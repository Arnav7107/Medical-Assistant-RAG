from fastapi import APIRouter, Form
from services.embeddings import get_embedding
from services.vectorstore import query_vector
from services.llm import generate_answer

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    # Step 1: Embed question
    query_emb = get_embedding(question)

    # Step 2: Query Pinecone
    matches = query_vector(query_emb, top_k=5)
    context = "\n\n".join([m.metadata.get("text", "") for m in matches])

    # Step 3: Generate answer
    answer = generate_answer(question, context)

    # Step 4: Return with sources
    sources = [m.metadata for m in matches]
    return {"response": answer, "sources": sources}
