# **Contextual Chatbot with Document Parsing**

This repository contains a contextual chatbot application designed to parse and analyze documents (PDF/Word), provide intelligent answers based on the context of the uploaded documents, and expose REST APIs using FastAPI. The solution uses LangChain for retrieval-based QA and integrates embeddings with Ollama.

## **Features**

- **Document Parsing**: Upload PDF or Word documents, which are parsed and split into chunks for processing.
- **Contextual Chatbot**: Answer user questions based on the context of the uploaded documents.
- **Semantic Search**: Retrieve the most relevant chunks using FAISS and semantic similarity search.
- **Open-Source Embeddings**: Utilize Ollama's `all-minilm` embeddings.
- **FastAPI**: Expose APIs for document upload and querying.
- **Streamlit Frontend**: Interactive frontend to upload documents and ask questions.
- **MLOps Integration**: Modular design for future pipeline enhancements, monitoring, and retraining.

---

## **Technologies Used**

- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Embeddings**: Ollama (`all-minilm`)
- **Document Store**: FAISS
- **QA Framework**: LangChain
- **Others**: matplotlib, scikit-learn for EDA and visualization.

---

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### **2. Set Up the Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Run the Backend**

Start the FastAPI server to expose the REST APIs.

```bash
cd backend
uvicorn app:app --reload
```

APIs will be available at `http://127.0.0.1:8000`.

### Running Tests

To run the tests and evaluate the chatbot performance:

```bash
python -m pytest --capture=no tests/test_evaluation.py
```


### **2. Run the Frontend**

Launch the Streamlit application.

```bash
cd frontend
streamlit run app.py
```

The Streamlit app will be available at `http://localhost:8501`.

---

## **Endpoints**

### **1. Upload Document**

**Endpoint**: `/upload/`  
**Method**: POST  
**Description**: Upload a document to parse and store embeddings.

**Request**:
```bash
curl -X POST -F "file=@your_document.pdf" http://127.0.0.1:8000/upload/
```

**Response**:
```json
{
  "message": "Document uploaded and processed successfully!"
}
```

### **2. Query Document**

**Endpoint**: `/query/`  
**Method**: POST  
**Description**: Query the document for contextual answers.

**Request**:
```json
{
  "question": "What is the main topic of the document?"
}
```

**Response**:
```json
{
  "answer": "The document discusses..."
}
```

---

## **Folder Structure**

```plaintext
.
├── backend/                      # Backend APIs and logic
│   ├── app.py                    # Main FastAPI application
│   ├── requirements.txt          # Backend dependencies
│   └── __init__.py
├── frontend/                     # Streamlit application
│   ├── app.py                    # Streamlit frontend
│   └── requirements.txt          # Frontend dependencies
├── src/                          # Core logic and utilities
│   ├── embeddings/               # Embedding generation
│   ├── preprocessing/            # Document loading and text splitting
│   ├── vector_store/             # FAISS vector store logic
│   ├── qa_pipeline/              # LangChain QA pipeline
│   ├── evaluation/               # EDA and visualization scripts
│   ├── utils/                    # Logger and constants
├── .gitignore                    # Ignored files and folders
├── README.md                     # Project documentation
└── requirements.txt              # Global dependencies
```

---

## **Development Notes**

- Ensure that the `FAISS` index is rebuilt after uploading a new document.
- Modify the `Ollama` model in `src/embeddings/ollama_embedding.py` if a different embedding model is preferred.
- Update the `retriever_k` parameter in the QA pipeline for more or fewer document chunks during retrieval.

---

## **Future Enhancements**

- Add user authentication for secure access.
- Deploy the chatbot to cloud platforms (AWS, GCP, or Azure).
- Integrate monitoring and retraining pipelines using MLOps tools.
- Expand support for multi-language document processing.

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add feature-name"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Contact**

For any questions or suggestions, feel free to open an issue or contact me at:  
📧 **jeet.undaviya20017@gmail.com**

--- 