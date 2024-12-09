import sys
import pytest
import os
import faiss
import numpy as np

# Adding folder path to sys.path
sys.path.append('/Users/jeetundaviya/Documents/rag_chatbot')

from src.evaluation.evaluate import evaluate_chatbot_performance
from src.vector_store.faiss_store import load_faiss_store
from src.preprocessing.document_loader import load_document
from src.preprocessing.text_splitter import split_document
from src.vector_store.faiss_store import create_faiss_store, save_faiss_store

def test_chatbot_evaluation():
    # Example usage to create FAISS index from the test PDF file
    file_path = "/Users/jeetundaviya/Documents/Projects/rag_chatbot/src/evaluation/Jeet_Undaviya_Resume.pdf"  # Your test file
    faiss_index_path = create_faiss_index_from_file(file_path)
    print(f"FAISS index saved to: {faiss_index_path}")

    # Load the FAISS index
    vector_store = load_faiss_store("src/vector_store/faiss_index.bin")
    
    # Run evaluation
    report = evaluate_chatbot_performance(vector_store, test_file="src/evaluation/test_dataset.json")
    print(report)
    # Assert metrics
    assert report["average_judgment_score"] >= 0.7, "Accuracy should be at least 70%"
    assert report["average_latency"] < 2, "Latency should be under 2 seconds"
    assert report["fallback_rate"] <= 0.2, "Fallback rate should be 20% or less"

def create_faiss_index_from_file(file_path):
    # Load the document using your existing loader function
    documents = load_document(file_path)

    # Split the document into chunks (you can use your custom text splitter)
    chunks = split_document(documents)

    # Create FAISS index from the embeddings
    faiss_index = create_faiss_store(chunks)

    # Save the FAISS index to a binary file
    faiss_index_path = "src/vector_store/faiss_index.bin"
    save_faiss_store(faiss_index, faiss_index_path)

    return faiss_index_path

