from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload_pdfs import router as upload_router
from routes.ask_question import router as ask_router

app = FastAPI(title="Medical Assistant RAG")

# CORS (so Streamlit frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(upload_router)
app.include_router(ask_router)
