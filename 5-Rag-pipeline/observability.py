import logging
from prometheus_client import Counter, Histogram, start_http_server

# --------------------------------------------------------
# metrics setup
# --------------------------------------------------------
REQUEST_COUNTER = Counter("genai_requests_total", "Total number of requests to the RAG pipeline")
LLM_LATENCY = Histogram("genai_llm_latency_ms", "Latency of LLM calls in milliseconds")
RETRIEVAL_LATENCY = Histogram("genai_retrieval_latency_ms", "Latency of context retrieval in milliseconds")

def log(question, model_input, model_output, user_id=None):
    REQUEST_COUNTER.inc()
    # Here you would typically log to a file or monitoring system
    logging.info("number of requests incremented: {REQUEST_COUNTER}")
    logging.info(f"question: {question}")
    logging.info(f"model_input: {model_input}")
    logging.info(f"model_output: {model_output}")
    if user_id:
        logging.info(f"user_id: {user_id if user_id else 'anonymous'}")

def record_metrics(metric_name, value):
    if metric_name == "genai_llm_latency_ms":
        LLM_LATENCY.observe(value)
    elif metric_name == "genai_retrieval_latency_ms":
        RETRIEVAL_LATENCY.observe(value)


# Start the Prometheus metrics server
def start_metrics_server(port=8000):
    start_http_server(port)
    logging.info(f"Prometheus metrics server started running at http://localhost:{port}/metrics")
        
