from typing import Optional
import uvicorn
import time
import uuid
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
handlers = [logging.FileHandler('rag_pipeline.log'), logging.StreamHandler()]
)

from retrieval import retrieve_context
from router import build_prompt
from llm_client import call_llm
from cache_store import get, set
from postprocess import secured_output
from guardrails import apply_guardrails
from observability import start_metrics_server, log, record_metrics
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RAG Pipeline API",
    description="A FastAPI service for the RAG pipeline",
    version="1.0.0"
)

# enable cors for frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ask request and response models
# purpose of these models is to define the structure of the request and response payloads for the /ask endpoint.
class AskRequest(BaseModel):
    question: str
    user_id: str | None = None
    
class AskResponse(BaseModel):
    answer: str
    request_id: str | None = None
    



# --------------------------------------------------------
# RAG PIPELINE MAIN MODULE
# --------------------------------------------------------
def run_rag_pipeline(question: str) -> str:
    logging.info(f"Received question: {question}")
    print(f"Received question: {question}")
    
    # step 0: check the cache
    cached_answer = get(question)
    if cached_answer:
        logging.info("Cache hit. Returning cached answer.")
        print("Cache hit. Returning cached answer.")
        return cached_answer
    print("Cache miss. Processing question.")

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
    answer = secured_output(answer)
    logging.info(f"Raw answer: {answer}")
    print(f"Raw answer: {answer}")
    answer = apply_guardrails(answer)
    logging.info(f"Guardrails applied answer: {answer}")
    print(f"Guardrails applied answer: {answer}")
    
    # Step 4: Cache the answer
    set(question, answer)
    logging.info("Answer cached for future requests.")
    log(question, prompt, answer) # Log the request and response

    return answer

# --------------------------------------------------------
# FASTAPI routes
# --------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    request_id = str(uuid.uuid4())
    answer = run_rag_pipeline(request.question,)
    print(f"response id: {answer}")
    print(type(answer))
    print(f"request id: {request_id}")
    print(type(request_id))
    return AskResponse(answer=answer, request_id=request_id)

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RAG Pipeline is healthy"}

@app.get("/")
def root():
    return {"message": "Welcome to the RAG Pipeline API", "metrics_endpoint": "/metrics", "health_endpoint": "/health"}

if __name__ == "__main__":
    start_metrics_server()
    uvicorn.run(app, host="0.0.0.0", port=8002)
    