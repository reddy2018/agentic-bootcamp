from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate


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


template = PromptTemplate.from_template(
    """Answer the following quesiton using the context below.
    If the question cannot be answered using the context, say "I don't know"."
    Context: {context}
    Question: {question}
    Answer: """
)

template_object = template.invoke(
    {"context": "Sweden is a country in Northern Europe.",
    "question":"What is the capital of Sweden?"}
)

# human_msg = HumanMessage(content="What is the capital of Sweden?")
# system_msg = SystemMessage(content="You are a helpful translator. Translate the user sentence to swedish.")

# message = [human_msg, system_msg]

question = llm.invoke(template_object)
response = llm.invoke(template_object)

print("Response from gpt-4o:")
print(question)
print(response)