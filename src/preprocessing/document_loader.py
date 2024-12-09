from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import filetype

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_docx(file_path: str):
    loader = Docx2txtLoader(file_path)
    return loader.load()

def load_document(file_path: str):
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.docx'):
        return load_docx(file_path)
    else:
        # Detect file type based on content
        kind = filetype.guess(file_path)
        if kind is None:
            raise ValueError("Cannot determine file type")
        
        if kind.mime == 'application/pdf':
            return load_pdf(file_path)
        elif kind.mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {kind.mime}")