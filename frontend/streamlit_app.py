import streamlit as st
import requests
import os
from uuid import uuid4

st.set_page_config(page_title="Jubilant's Medical AI", layout="centered")
st.title("JUBILANT'S AI MEDICAL ASSISTANT")
st.caption("Upload docs → Ask questions → Generate reports")

# Session persistence – keeps chat across reruns
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploaded_files = st.file_uploader(
    "Upload Medical Documents",
    type=["pdf", "docx", "xlsx", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Supports PDFs, Word, Excel, and images with OCR"
)

document_paths = []
if uploaded_files:
    files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
    with st.spinner("Uploading..."):
        response = requests.post("http://localhost:8000/upload", files=files)
    if response.status_code == 200:
        st.success(f"Uploaded: {', '.join([f.name for f in uploaded_files])}")
        document_paths = response.json()["files"]
    else:
        st.error("Upload failed")

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Ask a question or say: Generate report with Introduction and Summary")

if query and document_paths:
    payload = {"query": query, "documents": document_paths, "session_id": st.session_state.session_id}
    with st.spinner("Processing..."):
        response = requests.post("http://localhost:8000/query", json=payload)

    if response.status_code == 200:
        if "application/pdf" in response.headers.get("content-type", ""):
            st.session_state.chat_history.append({"role": "assistant", "content": "Report ready!"})
            st.download_button("Download Report PDF", response.content, "report.pdf", "application/pdf")
        else:
            data = response.json()
            resp = data.get("response", "")
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
            st.rerun()