# app/rag/rag_pipeline.py
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from loguru import logger

class QAAgent:
    def __init__(self, vectorstore):
        logger.info("Initializing QAAgent")
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Use a valid embedding model
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore.embedding_function = self.embeddings.embed_query

        self.llm = ChatOllama(model="llama3:8b", temperature=0.1)

        prompt = ChatPromptTemplate.from_template(
            """Use the following context to answer the query. Be grounded in the documents.
            Context: {context}
            History: {history}
            Query: {query}
            Answer:"""
        )

        # Adjust the chain to handle history and query correctly
        chain = (
            RunnableParallel(
                context=lambda x: self.retriever.invoke(x.get("query", x) if isinstance(x, dict) else x),
                query=lambda x: x.get("query", x) if isinstance(x, dict) else x,
                history=lambda x: x.get("history", []) if isinstance(x, dict) else []
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Session-based memory store
        self.memory_store = {}

        self.runnable_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=self.get_session_history,
            input_messages_key="query",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str):
        if session_id not in self.memory_store:
            self.memory_store[session_id] = InMemoryChatMessageHistory()
        return self.memory_store[session_id]

    def answer(self, query, session_id="default"):
        logger.info(f"Answering query: {query} for session {session_id}")
        response = self.runnable_with_history.invoke(
            {"query": query},
            config={"configurable": {"session_id": session_id}},
        )
        return response