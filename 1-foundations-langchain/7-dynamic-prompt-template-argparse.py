#prompt | llm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse

load_dotenv()  # take environment variables from .env.

args = argparse.ArgumentParser(description="Dynamic Prompt Template with argparse")
args.add_argument("--style", type=str, required=True, help="style of the explainer")
args.add_argument("--topic", type=str, required=True, help="topic of the explainer  ")
args.add_argument("--audience", type=str, required=True, help="target audience")  
args.add_argument("--length", type=str, required=True, help="length of the explainer")
parsed_args = args.parse_args()

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
        ("system", "your are in expert assistant. adapt explanations to the audience and style."
         "prefer short sentences and concrete examples when helpful."),
        ("user",
            """write a {style} explainer on the topic of {topic} for a {audience}.
            keep it {length} long.
            Format: \n
            opener: 1-2 lines \n
            core: 3-5 bullet points \n
            bottem line: single sentence starting with 'The bottom line is' \n
            """)
    ]
)

chain = prompt | llm

variables = {
    "style": parsed_args.style, 
    "topic": parsed_args.topic,
    "audience": parsed_args.audience,   
    "length": parsed_args.length
}
response = chain.invoke(variables)
print("Response from gpt-4o:")  
print(response.content)