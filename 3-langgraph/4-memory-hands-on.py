from typing import TypedDict, List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# MemorySaver: is the checkpointer used to persist state/memory across runs (the library determines backend; if no args, it may be in-memory or default file-based depending on version).
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

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

# Define the graph state structure
# this defines what kind of data your graph state should hold
class GraphState(TypedDict):
    """
    input: latest user input
    summary: rolling summary of the conversation
    messages: ordered list of all messages exchanged # chart history
    """
    input: str
    summary: Optional[str]
    messages: List[BaseMessage]
    
# ingest_user
def ingest_user(state: GraphState) -> GraphState:
    messages = list(state.get("messages", []))
    user_text = state["input"]
    messages.append(HumanMessage(content=user_text))
    state["messages"] = messages
    return state

# at the starting
# messages = []
# messages = [HumanMessage(content="user input here")]
# chat node
def chat(state: GraphState) -> GraphState:
    messages = list(state.get("messages", []))
    summary = state.get("summary", "The conversation is starting.")
    
    effective_message: List[BaseMessage] = [] # previous summary as system message and it should be prepended to the messages
    if summary:
        effective_message.append(SystemMessage(content=f"Summary of conversation so far: {summary}"))
    effective_message.extend(messages)
    
    response = llm.invoke(effective_message)
    messages.append(AIMessage(content=response.content))
    state["messages"] = messages
    return state



summary_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an expert at creating concise summaries of conversations."),
    HumanMessage(content="Given the following conversation history, provide a concise summany by focusing on key facts, preference and decisions: \n{conversation_history}")
])
THRESHOLD = 2 # minimum number of messages to trigger summary update

# update summary node
def summarize_if_long(state: GraphState) -> GraphState:
    messages = list(state.get("messages", []))
    summary = state.get("summary", "")
    
    # check if message length exceeds threshold
    if len(messages) < THRESHOLD:
        return state  # no update needed
    
    # generate new summary
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    prompt = summary_prompt.format_prompt(conversation_history=conversation_text)
    response = llm.invoke(prompt.to_messages())
    state["summary"] = response.content
    return state

# Build the graph workflow
# [START] → [ingest_user] → [chat] → [summarize_if_long] → [END]  
graph = StateGraph(GraphState)
graph.add_node("ingest_user", ingest_user)
graph.add_node("chat", chat)
graph.add_node("summarize_if_long", summarize_if_long)
graph.add_edge(START, "ingest_user")
graph.add_edge("ingest_user", "chat")
graph.add_edge("chat", "summarize_if_long")
graph.add_edge("summarize_if_long", END)


checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

def run_turn(user_input: str, thread_id: str):
    # initial state
    result = app.invoke(
        {"input": user_input},
        config={"configurable": {"thread_id": thread_id}}
    )
    return result

if __name__ == "__main__":
    thread_id = "conversation_001"
    s1 = run_turn("Hello! Can you tell me about langgraph?", thread_id)
    print("after turn 1:", s1)
    print(s1["messages"][-1].content)
    
    # turn 2
    s2 = run_turn("Can you give me an example use case?", thread_id)
    print("after turn 2:", s2)
    print(s2["messages"][-1].content)
    
    # turn 3
    s3 = run_turn("How does memory management work?", thread_id)
    print("after turn 3:", s3)
    print(s3["messages"][-1].content)
    
    # show the persistent storage content
    print("\nPersistent storage content:")
    print("summary:",s3.get("summary"))
    
    # export the graph visualization
    app.get_graph().draw_mermaid_png(output_file_path="memory_hands_on_graph.png")