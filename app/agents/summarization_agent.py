from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from loguru import logger

class SummarizationAgent:
    def __init__(self):
        logger.info("SummarizationAgent initialized – only used when user asks for summary")
        self.llm = ChatOllama(model="llama3:8b", temperature=0.1)  # Low temp = factual

    def summarize(self, text, length="concise"):
        """Generates concise summary – preserves medical accuracy."""
        logger.info(f"Summarizing text ({len(text)} chars) → {length}")
        prompt = PromptTemplate(
            template="Summarize this medical text in a {length} way. Do not add or omit facts:\n\n{text}",
            input_variables=["length", "text"]
        )
        response = self.llm.invoke(prompt.format(length=length, text=text[:4000]))  # Truncate if too long
        summary = response.content.strip()
        logger.debug(f"Summary generated: {summary[:100]}...")
        return summary