# example memory_file.py
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# Initialize the ChatOpenAI model with desired parameters
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",
    # base_url="...",
    # organization="...",
    # other params...
)

# Define the node function that uses the LLM
def llm_node(state: dict) -> dict: # state dictonary containes all current state information
    user_input = state.get("input", "")
    initial_response = state["response"]
    print("Initial response in llm_node:", initial_response)
    response = llm.invoke(user_input)
    return {"response": response.content}

# Define the graph state structure
# this defines what kind of data your graph state should hold
class GraphState(TypedDict):
    input: str
    response: str


# Build the graph workflow
# [START] → [llm_node] → [END]
graph = StateGraph(GraphState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

# prepare the input and compile the graph
input_state = {"input": "Explain langgraph in simple terms.", "response": "This is a placeholder response."}
app = graph.compile()

# run the graph and get the result
result = app.invoke(input_state)
# Execute the graph with the input state and print the output
# Export the graph visualization
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print("Final output:", result["response"])
