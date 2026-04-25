import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import process_pdf, ask_question, is_pdf_processed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "Ask-My-PDF API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        process_pdf(pdf_path)
        # Save cache marker
        os.makedirs("./chroma_db", exist_ok=True)
        with open("./chroma_db/.pdf_source", "w") as f:
            f.write(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"'{file.filename}' uploaded and processed successfully"}

@app.post("/ask")
def ask(request: QuestionRequest):
    if not is_pdf_processed():
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet. Please upload a PDF first."
        )
    result = ask_question(request.question)
    return result