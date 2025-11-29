import time
import argparse
import logging

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
handlers = [logging.FileHandler('rag_pipeline.log'), logging.StreamHandler()]
)

from retrieval import retrieve_context
from router import build_prompt
from llm_client import call_llm

def run_rag_pipeline(question: str) -> str:
    logging.info(f"Received question: {question}")

    # Step 1: Retrieve context
    start_time = time.time()
    context = retrieve_context(question)
    end_time = time.time()
    retrieval_time = end_time - start_time
    retrieval_latency = int(retrieval_time * 1000)  # Convert to milliseconds)
    logging.info(f"Context retrieved in {retrieval_latency} milliseconds")

    # Step 2: Build prompt
    model_name, prompt = build_prompt(question, context)
    logging.info(f"Prompt built for model {model_name}")
    logging.info(f"Prompt: {prompt}")

    # Step 3: Call LLM
    start_time = time.time()
    answer = call_llm(model_name, prompt)
    end_time = time.time()
    llm_time = end_time - start_time
    llm_latency = int(llm_time * 1000)  # Convert to milliseconds
    logging.info(f"LLM response received in {llm_latency} milliseconds")

    return answer

if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser(description="Run RAG Pipeline")
    #parser.add_argument("--question", type=str, required=True, help="The question to ask the RAG pipeline")
    #args = parser.parse_args()
    
    # create a chat interface to ask questions
    question = input("Enter your question: ")
    args = argparse.Namespace(question=question)
    

    answer = run_rag_pipeline(args.question)
    print("Answer:", answer)