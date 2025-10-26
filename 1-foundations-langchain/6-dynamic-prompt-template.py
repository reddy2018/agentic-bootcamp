from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
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
        ("system", "you are in expert assistant. adapt explanations to the audience and style."
         "prefer short sentences and concrete examples when helpful."),
        ("user", 
         """write a {style} explainer on the topic of {topic} for a {audience}.
         keep it {length} long.
         Format: \n
         opener: 1-2 lines \n
         core: 3-5 lines \n
         bottem line: single sentence starting with 'The bottom line is' \n
         """)
    ]
)

chain = prompt | llm
variables = {
    "style": "casual", 
    "topic": "quantum computing",
    "audience": "high school student",
    "length": "150 words"
}
response = chain.invoke(variables)
print("Response from gpt-4o:")
print(response.content)