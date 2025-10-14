from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()  # take environment variables from .env.

# https://python.langchain.com/docs/integrations/chat/openai/
# https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#langchain_google_genai.chat_models.ChatGoogleGenerativeAI

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

llm_v2 = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",
    # base_url="...",
    # organization="...",
    # other params...
)

message = [
    (
        "system",
        "You are a helpful translator. Translate the user sentence to swedish.",
    ),
    ("human", "I love programming."),
]


response = llm.invoke(message)
response_v2 = llm_v2.invoke(message)

print("Response from gpt-4.1-nano:")
print(response_v2)

print("Response from gpt-4o:")
print(response)