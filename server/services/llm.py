import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

def generate_answer(question: str, context: str) -> str:
    """Generate answer from Gemini using retrieved context"""
    prompt = f"""
You are a medical assistant. Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:
"""
    response = llm.invoke(prompt)
    return response.content
