from langchain_ollama import OllamaEmbeddings

def get_ollama_embedding_model(model: str = "all-minilm"):
    """
    Returns an instance of LangChain's OllamaEmbeddings.
    """
    return OllamaEmbeddings(model=model)

def get_ollama_embeddings(chunks:list):
    embeddings_model = get_ollama_embedding_model()
    return embeddings_model.embed_documents(chunks)
