#prompt | llm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse

load_dotenv()  # take environment variables from .env.
args = argparse.ArgumentParser(description="Travel Recommender with argparse")
args.add_argument("--city", type=str, required=True, help="destination city")  
args.add_argument("--budget", type=str, required=True, help="travel budget")
args.add_argument("--days", type=int, required=True, help="number of days for the trip")
args.add_argument("--traveler_type", type=str, required=True, help="type of traveler (e.g., solo, family, couple)")
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
        ("system", "You are a travel recommendation assistant. Provide personalized travel suggestions based on user preferences."),
        ("user",    
            """Recommend a travel destination based on the following preferences:
            - Destination city: {city}
            - Budget: {budget}
            - days: {days}
            - type of traveler: {traveler_type}
            Format your response as follows:
            Opener: 1-2 lines
            itinerary: day by day plan matching the number of days
            tips: 2-3 short suggestions tailored to budget and traveler type
            """)
    ]
)   
chain = prompt | llm

variables = {
    "city": parsed_args.city,   
    "budget": parsed_args.budget,
    "days": parsed_args.days,
    "traveler_type": parsed_args.traveler_type
}
response = chain.invoke(variables)
print("Response from gpt-4o:")
print(response.content)
