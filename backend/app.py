import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import tempfile
import shutil
import traceback

# Adding folder path to sys.path
sys.path.append('/Users/jeetundaviya/Documents/rag_chatbot')

from src.utils.logger import setup_logger
from src.preprocessing.document_loader import load_document
from src.preprocessing.text_splitter import split_document
from src.embeddings.ollama_embedding import get_ollama_embeddings
from src.vector_store.faiss_store import create_faiss_store, query_faiss_index
from src.qa_pipeline.langchain_qa import setup_langchain_qa,query_qs

app = FastAPI()

# Set up logger
log_file = "app_logs.log"  # Define log file path
logger = setup_logger(name="my_app_logger", log_file=log_file)

# Global variables
faiss_index = None
documents = []

class QueryRequest(BaseModel):
    question: str

def process_uploaded_document(file: UploadFile):
    global faiss_index, documents
    # Temporary file to hold the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        temp_file.close()
        file_path = temp_file.name

    # Load the document using the appropriate loader
    try:
        documents = load_document(file_path)
        os.remove(file_path)  # Cleanup the temp file
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise HTTPException(status_code=400, detail="Error loading document")

    # Split the document into chunks
    try:
        chunks = split_document(documents)
    except Exception as e:
        logger.error(f"Error splitting document: {e}")
        raise HTTPException(status_code=400, detail="Error splitting document")
    
    # Create a FAISS index
    try:
        faiss_index = create_faiss_store(chunks)
        logger.info("FAISS index created successfully.")
    except Exception as e:
        logger.error(f"Error creating FAISS store: {e}")
        raise HTTPException(status_code=400, detail="Error creating FAISS store")

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    try:
        process_uploaded_document(file)
        return {"message": "Document uploaded and processed successfully!"}
    except HTTPException as e:
        raise e  # Forward the error with status code and detail

@app.post("/query/")
async def query_document(request: QueryRequest):
    global faiss_index
    if not faiss_index:
        raise HTTPException(status_code=400, detail="No document uploaded")

    try:
        # Set up the Langchain QA chain
        qa_chain = setup_langchain_qa(faiss_index, model="llama3.2", retriever_k=3)
        logger.info('[+] QA Chain Successfuly created !')

        logger.info(f'[+] request.question : {request.question} !')

        # Get the answer using the QA chain
        answer = query_qs(request.question,faiss_index)
        logger.info('[+] QA Chain Successfuly invoked !')
        logger.info({"answer": answer})
        return {"answer": answer.content}
    
    except Exception as e:
        logger.error(f"Error querying document: {e} [Exception] :- {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error processing query")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
