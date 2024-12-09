from langchain_text_splitters import CharacterTextSplitter

def split_document(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def split_document_to_text(documents, chunk_size=512, chunk_overlap=50):
    docs = split_document(documents, chunk_size=512, chunk_overlap=50)
    return [doc.page_content for doc in docs]