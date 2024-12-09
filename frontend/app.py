import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("Document Q&A Chatbot")

# Upload Document
st.subheader("Step 1: Upload Document")
uploaded_file = st.file_uploader("Choose a document (PDF/Text)", type=["pdf", "txt"])

if uploaded_file is not None:
    # Send document to backend API for processing
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{BACKEND_URL}/upload/", files=files)
    if response.status_code == 200:
        st.success("Document uploaded successfully!")
    else:
        st.error("Failed to upload the document.")

# Ask Question
st.subheader("Step 2: Ask a Question")
question = st.text_input("Enter your question:")

if question:
    # Send question to backend API for answering
    response = requests.post(f"{BACKEND_URL}/query/", json={"question": question})
    if response.status_code == 200:
        answer = response.json().get("answer")
        if answer:
            st.write(f"Answer: {answer}")
        else:
            st.write("Sorry, I don't know the answer.")
    else:
        st.write("Failed to get an answer.")
