from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()

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


def llm_node(state: dict) -> dict:
    user_input = state.get("input", "")
    initial_response = state["response"]
    print("Initial response in llm_node:", initial_response)
    response = llm.invoke(user_input)
    return {"response": response.content}

class GraphState(TypedDict):
    input: str
    response: str   
    
# Define the state graph with a start node, an LLM processing node, and an end node
graph = StateGraph(GraphState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

input_state = {"input": "Explain langgraph in simple terms.", "response": "This is a placeholder response."}
app = graph.compile()

# Execute the graph with the input state and print the output
result = app.invoke(input_state)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print("Final output:", result["response"])


