from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from app.agents.document_loader import DocumentLoaderAgent
from app.rag.rag_pipeline import QAAgent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.summarization_agent import SummarizationAgent
from app.agents.report_assembly_agent import ReportAssemblyAgent
from loguru import logger
import re
from collections import defaultdict

class AgentState(TypedDict):
    query: str
    documents: List[str]
    vectorstore: any
    sections: dict
    response: any
    session_id: str
    classification: str
    memory: Optional[ConversationBufferMemory]

@tool
def load_docs(documents: List[str]):
    """Tool: Loads and indexes documents into FAISS."""
    loader = DocumentLoaderAgent()
    return {"vectorstore": loader.load_documents(documents)}

@tool
def handle_qa(query: str, vectorstore: any, session_id: str):
    """Tool: Answers questions using RAG + conversation memory."""
    qa = QAAgent(vectorstore)
    return {"response": qa.answer(query, session_id)}

@tool
def extract_content(path: str, section: str):
    """Tool: Pulls exact content (text/table/image) from a document."""
    extractor = ExtractionAgent()
    if "table" in section.lower():
        return extractor.extract_table(path)
    elif "image" in section.lower():
        return extractor.extract_image(path)
    else:
        return extractor.extract_text(path, section)

@tool
def summarize_content(text: str):
    """Tool: Summarizes only when explicitly requested."""
    summarizer = SummarizationAgent()
    return summarizer.summarize(text)

@tool
def assemble_report(sections: dict):
    """Tool: Compiles final PDF report."""
    assembler = ReportAssemblyAgent()
    return {"response": assembler.assemble_report(sections)}

class Orchestrator:
    def __init__(self):
        self.llm = ChatOllama(model="llama3:8b", temperature=0.3)
        self.tools = [load_docs, handle_qa, extract_content, summarize_content, assemble_report]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Per-session memory – critical for multi-turn conversations
        self.memory_store = defaultdict(lambda: ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        ))

        # Classifier: decides QA vs Report
        classify_prompt = PromptTemplate(
            template="Classify: {query}\nIs it 'QA' (question) or 'Report' (generate report)? Answer only 'QA' or 'Report'.",
            input_variables=["query"]
        )
        self.classifier = classify_prompt | self.llm

        def classify(state: AgentState):
            classification = self.classifier.invoke({"query": state["query"]}).content.strip()
            logger.info(f"Query classified as: {classification}")
            return {
                "classification": classification,
                "memory": self.memory_store[state["session_id"]]
            }

        def route(state: AgentState):
            return "report_flow" if state["classification"] == "Report" else "qa_flow"

        def qa_flow(state: AgentState):
            load_result = load_docs.invoke({"documents": state["documents"]})
            state["vectorstore"] = load_result["vectorstore"]

            # Reconstruct chat history for context
            memory = state["memory"]
            history = memory.load_memory_variables({})["chat_history"]
            context = "\n".join([f"{m.type}: {m.content}" for m in history[-4:]])  # Last 2 turns
            full_query = f"{context}\nUser: {state['query']}" if context else state["query"]

            result = handle_qa.invoke({
                "query": full_query,
                "vectorstore": state["vectorstore"],
                "session_id": state["session_id"]
            })
            state["response"] = result["response"]
            memory.save_context({"input": state["query"]}, {"output": state["response"]})
            return state

        def report_flow(state: AgentState):
            # Parse requested sections from query
            match = re.search(r"with\s+(.+)", state["query"], re.IGNORECASE)
            sections = [s.strip() for s in (match.group(1).split(",") if match else ["Introduction", "Summary"])]
            state["sections"] = {}

            for section in sections:
                for doc in state["documents"]:
                    content = extract_content.invoke({"path": doc, "section": section})
                    if content and "summary" in section.lower():
                        content = summarize_content.invoke({"text": content})
                    if content:
                        state["sections"][section] = content
                        break

            result = assemble_report.invoke({"sections": state["sections"]})
            state["response"] = result["response"]
            return state

        # Build LangGraph workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("classify", RunnableLambda(classify))
        workflow.add_node("qa_flow", RunnableLambda(qa_flow))
        workflow.add_node("report_flow", RunnableLambda(report_flow))
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges("classify", RunnableLambda(route))
        workflow.add_edge("qa_flow", END)
        workflow.add_edge("report_flow", END)

        self.graph = workflow.compile()
        logger.success("Orchestrator initialized with agentic workflow")

    def invoke(self, input_data):
        state = {**input_data, "sections": {}, "response": None}
        logger.info(f"Orchestrator invoked with query: {state['query'][:50]}...")
        return self.graph.invoke(state)