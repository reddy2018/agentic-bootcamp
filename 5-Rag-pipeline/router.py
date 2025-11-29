# it container both prompt and retrieval logic
# this is the place where you can have guardians or routers to decide which model or retrieval strategy to use
# you can also have a place where you can have the prompt engineering logic
from config import DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS
TEMPLATE = """You are a help full assistant. answer the questions as best as you can using the context provided.
{context_block}

Question: {question}

Answer:"""

def build_prompt(question: str, context: str) -> tuple[str]:
    context_block = f"Context:\n{context}" if context.strip() else "No relevant context found."
    return DEFAULT_MODEL, TEMPLATE.format(context_block=context_block, question=question)