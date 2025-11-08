from typing import TypedDict, Optional, List
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated
import numexpr
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage

# take environment variables from .env
load_dotenv()

@tool
def calculator_tool(query: str) -> str:
    """
    A simple calculator tool. input should be a mathematical expression.
    """
    
    return str(numexpr.evaluate(query))

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, calculator_tool]

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
).bind_tools(tools)


class state(TypedDict):
    """
    messages: user input messages
    """
    
    messages: Annotated[list, add_messages]
    
def llm_node(state: state) -> state:
    print("Messages passed to LLM:")
    print(state["messages"])
    result = llm.invoke(state["messages"])
    print ("Intermidiate LLM result:")
    print(result)
    return {"messages": result}

builder = StateGraph(state)
builder.add_node("model", llm_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")
builder.add_edge("model", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="6-tools-v1.png")

input = {
    "messages": [HumanMessage(content = "calculate 123 * 456"),]
}

input = {
    "messages": [HumanMessage(content = "Calculate the Fixed deposit interest for 100000 INR for 1 year at the current RBI repo rate?"),]
}

result = graph.invoke(input)
print("Final result:")
print(result)
