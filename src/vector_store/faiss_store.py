from langchain_community.vectorstores import FAISS
from src.embeddings.ollama_embedding import get_ollama_embedding_model

def create_faiss_store(chunks, model="all-minilm"):
    embeddings = get_ollama_embedding_model(model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def save_faiss_store(vector_store, file_path):
    vector_store.save_local(file_path)

def load_faiss_store(file_path, model="all-minilm"):
    embeddings = get_ollama_embedding_model(model)
    return FAISS.load_local(file_path, embeddings,allow_dangerous_deserialization=True)

def query_faiss_index(faiss_index, query, k=3):
    return faiss_index.similarity_search(query, k)