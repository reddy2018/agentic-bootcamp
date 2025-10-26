# lcel demo
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
load_dotenv()  # take environment variables from .env.

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

# lcel compositon

chain = template | llm

response = chain.invoke({"context": "the capital of Sweden is Stockholm.",
    "question":"What is the capital of India?"})

print("Response from gpt-4o:")
print(response)