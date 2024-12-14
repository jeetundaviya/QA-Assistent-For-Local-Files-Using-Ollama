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
uvicorn backend.app:app --reload
```

APIs will be available at `http://127.0.0.1:8000`.

### Running Tests

To run the tests and evaluate the chatbot performance:

```bash
python -m pytest --capture=no tests/test_evaluation.py
```

### Sample Output

```
================== test session starts ===================
platform darwin -- Python 3.11.4, pytest-8.3.4, pluggy-1.5.0
rootdir: /Users/jeetundaviya/Documents/Projects/rag_chatbot
plugins: anyio-4.7.0
collected 1 item                                         

tests/test_evaluation.py FAISS index saved to: src/vector_store/faiss_index.bin
================================ Human Message =================================

Where is Jeet currently employed at ??
================================== Ai Message ==================================

Jeet Undaviya is currently employed as a Software Engineer at Motorola Solutions in Bengaluru, Karnataka.
[+] Judgement Score :- 1.0
================================ Human Message =================================

Which technologies is Jeet proficient in?
================================== Ai Message ==================================

According to the provided context, Jeet Undaviya is proficient in:

1. Objective-C
2. C/C++
3. iOS Development (Swift)
4. Java
5. Python
6. Flutter/Dart
7. C#
8. Android
[+] Judgement Score :- 0.9
================================ Human Message =================================

What is Jeet's current role?
================================== Ai Message ==================================

Jeet Undaviya is currently a Software Engineer at Motorola Solutions in Bengaluru, Karnataka.
[+] Judgement Score :- 1.0
{'average_judgment_score': 0.9666666666666667, 'average_latency': 1.8116357326507568, 'fallback_rate': 0.0, 'total_queries': 3}

==================== warnings summary ====================
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============= 1 passed, 3 warnings in 7.60s ==============
```

### Evaluation Metrics

- **Average Judgment Score**: 0.97
- **Average Latency**: 1.81 seconds
- **Fallback Rate**: 0.0 (No fallback responses)
- **Total Queries**: 3

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
rag_chatbot/
│
├── src/
│   ├── embeddings/              # Embedding models and utilities
│   │   ├── ollama_embedding.py # Ollama embeddings
│   ├── evaluation/              # Performance evaluation
│   │   └── evaluate.py         # Evaluation logic
│   ├── preprocessing/           # Document preprocessing (loading, splitting)
│   │   ├── document_loader.py  # Document loading functions
│   │   └── text_splitter.py    # Text chunking for long documents
│   ├── qa_pipeline/             # Question-Answering pipeline
│   │   └── langchain_qa.py     # LangChain-based QA pipeline
│   ├── utils/                   # Utility functions (logging, constants)
│   │   ├── constants.py        # Constants used in the project
│   │   └── logger.py           # Logging configuration
│   └── vector_store/            # Vector store (FAISS)
│       ├── faiss_store.py      # FAISS indexing and querying
│       └── faiss_index.bin     # FAISS index file
│
├── tests/                       # Test cases for evaluation and other components
│   └── test_evaluation.py      # Test cases for chatbot evaluation
│
├── .gitignore                   # Git ignore file
├── LICENSE                      # License file
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## **Development Notes**

- Ensure that the `FAISS` index is rebuilt after uploading a new document.
- Modify the `Ollama` model in `src/embeddings/ollama_embedding.py` if a different embedding model is preferred.
- Update the `retriever_k` parameter in the QA pipeline for more or fewer document chunks during retrieval.

---

## MLOps Pipeline

The **MLOps Pipeline** for this project integrates model development, deployment, and performance monitoring to ensure a robust, reproducible, and efficient workflow. Below is the structure of the pipeline:

1. **Data Ingestion**:
   - Documents (e.g., PDF, Word) are uploaded and processed by the system. The text is extracted, preprocessed, and split into smaller chunks.
   - The documents are stored in a FAISS index to facilitate fast similarity-based searches.

2. **Model Training**:
   - **Ollama LLM** (or similar models) are used for question-answering. The model is used to process the retrieved documents based on user queries.

3. **Model Evaluation**:
   - A dedicated evaluation pipeline assesses the chatbot’s performance using predefined metrics like **accuracy**, **latency**, and **fallback rate**.
   - The evaluation also involves **human-judgment scoring** based on the answers provided by the model.
   - Results from the evaluation are recorded and used to adjust model parameters for improved performance.

4. **Deployment**:
   - The chatbot is deployed as a FastAPI application for serving real-time queries.
   - Continuous monitoring and logging ensure that the system operates smoothly and provides insights for performance improvements.

5. **Continuous Integration/Continuous Deployment (CI/CD)**:
   - **CI/CD pipelines** can be set up using tools like GitHub Actions, Jenkins, or GitLab CI to automatically test and deploy updates.
   - Ensure that new changes do not break the pipeline by running automated tests on every push.

6. **Performance Monitoring**:
   - **Latency** and **fallback rates** are regularly tracked to ensure the model performs within the desired thresholds.
   - Evaluation reports are used to monitor progress and make informed decisions about further optimizations.
   
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
