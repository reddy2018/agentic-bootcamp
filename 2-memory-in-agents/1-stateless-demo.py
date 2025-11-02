from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are concise assistant. answer breifly as possible."),
        ("user", "{user_input}")
    ]
)
chain = prompt | llm

print("Welcome to the Stateless Chatbot! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Existing the demo!")
        break

    variables = {
        "user_input": user_input
    }
    response = chain.invoke(variables)
    print("Bot:", response.content)