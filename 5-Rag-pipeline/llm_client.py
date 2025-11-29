from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import TEMPERATURE, MAX_TOKENS
from dotenv import load_dotenv
load_dotenv()
import os
import logging
from functools import lru_cache

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# initialize the chat LLM client
@lru_cache(maxsize=10) # cache up the results of the function to avoid re-initializing the client multiple times
def _get_chat_llm(model_name: str) -> ChatOpenAI:
    logging.info(f"Initializing LLM client with model: {model_name}")
    return ChatOpenAI(
        model_name=model_name,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY
    )

# function to call the LLM with a prompt and return the response
def call_llm(model_name: str, prompt: str) -> str:
    llm = _get_chat_llm(model_name)
    logging.info(f"calling the chart model for: {model_name} with prompt: {prompt}")
    response:AIMessage = llm.invoke(prompt) # invoking the LLM with the prompt with AIMessage response type
    return response.content