from langchain_openai import ChatOpenAI
#from langchain.memory import ConversationBufferMemory
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# difine a simple tool (can be any function)
# ---------------- Calculator tool -------------
@tool
def calculator_tool(expression: str) -> str:
    """Evaluates a math expression and return the result."""
    try:    
        result = eval(expression)
        return f"the result is {result}"
    except expression as e:
        return f"error evaluating expression: {e}"


# ---------- Initialize LLM ---------- 
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# create a memory object

#memory = conversationaBufferMemory(memory_key="chat_history", return_messages=True)
memory = MemorySaver()

# register tools

tools = [calculator_tool]

# create the agent with memory + tool

agent = create_agent(
    tools=tools,
    model=llm,
    checkpointer=memory,
)

# 6️⃣ Helper to extract the latest assistant message content
def get_last_message_content(response):
    """Extract content safely from the latest AIMessage."""
    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        if messages and hasattr(messages[-1], "content"):
            return messages[-1].content
    elif hasattr(response, "content"):
        return response.content
    return str(response)


# use the thread id to maintain conversation history
thread_id = "session-1"

response1 = agent.invoke({"messages": [{"role": "user", "content": "Hi, I'm Alex. Can you remember my name?"}]}, config={"thread_id": thread_id})
print(get_last_message_content(response1))

# Second message (agent recalls name)
response2 = agent.invoke({"messages": [{"role": "user", "content": "What’s my name?"}]}, config={"thread_id": thread_id})
print(get_last_message_content(response2))

# Third message (tool call)
response3 = agent.invoke({"messages": [{"role": "user", "content": "Can you calculate 25 * 12 + 7 using your calculator?"}]}, config={"thread_id": thread_id})
print(get_last_message_content(response3))