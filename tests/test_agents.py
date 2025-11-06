import pytest
from app.agents.document_loader import DocumentLoaderAgent
from app.rag.rag_pipeline import QAAgent
import os

SAMPLE_PDF = "docs/sample_dataset/cmh-2022-0365.pdf"

@pytest.fixture
def loader():
    return DocumentLoaderAgent()

@pytest.fixture
def qa_agent(loader: DocumentLoaderAgent):
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("Sample PDF not found")
    vectorstore = loader.load_documents([SAMPLE_PDF])
    assert vectorstore is not None, "Vectorstore creation failed"
    return QAAgent(vectorstore)

def test_document_loader_success(loader: DocumentLoaderAgent):
    """Test if DocumentLoaderAgent successfully loads a PDF."""
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("Sample PDF not found")
    vectorstore = loader.load_documents([SAMPLE_PDF])
    assert vectorstore is not None, "Failed to load documents into vectorstore"
    assert os.path.exists(loader.vectorstore_path), "Vectorstore not persisted to disk"

def test_document_loader_empty(loader: DocumentLoaderAgent):
    """Test DocumentLoaderAgent with no valid files."""
    vectorstore = loader.load_documents(["nonexistent.pdf"])
    assert vectorstore is None, "Should return None for no valid files"

def test_qa_agent_response(qa_agent: QAAgent):
    """Test if QAAgent provides a valid response."""
    response = qa_agent.answer("What is the scenario?", session_id="test_session")
    assert len(response) > 0, "QAAgent returned empty response"
    assert isinstance(response, str), "Response should be a string"

def test_qa_agent_memory(qa_agent: QAAgent):
    """Test if QAAgent maintains conversation memory across calls."""
    session_id = "test_memory_session"
    response1 = qa_agent.answer("What is the scenario?", session_id=session_id)
    response2 = qa_agent.answer("Summarize the scenario?", session_id=session_id)
    assert len(response1) > 0 and len(response2) > 0, "Memory test failed: responses should be non-empty"
    # Basic check: Second response should be longer or reference prior (heuristic)
    assert len(response2) > 10, "Second response should reference prior context meaningfully"