from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import argparse
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser
import json

load_dotenv()  # take environment variables from .env.

class RestaurantInfo(BaseModel):
    name: Optional[str] = Field(..., description="The name of the restaurant")
    cuisine: Optional[str] = Field(..., description="The type of cuisine the restaurant serves, e.g., Italian, Chinese, Mexican")
    location: Optional[str] = Field(None, description="The location of the restaurant")
    
    rating : Optional[float] = Field(None, description="Average rating of the restaurant (0.0 to 5.0) allow decimal values")
    price_range : Optional[str] = Field(None, description="Price range of the restaurant, e.g., low, medium, high")
    
args = argparse.ArgumentParser(description="Restaurant Information Extractor")
args.add_argument("--text", type=str, required=True, help="A paragraph describing the restaurant (e.g., 'Loved dinner at Trattoria Bella in central Rome… a bit pricey… 4.5/5.')")
args.add_argument("--max_retries", type=int, default=2, help="Maximum number of retries for the LLM")
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


parser = PydanticOutputParser(pydantic_object=RestaurantInfo)
format_instructions = parser.get_format_instructions()
print("Format Instructions:")
print(format_instructions)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """you are an information extraction assistant. extract clean restaurant information from user input.
            if the field is unknown for the provided text, set it to null
            and never fabricate the facts.
            """),
        ("user",
            """
            Extract restaurant information from the text and format it as per the instructions below:
            {text}\n
            Follow the format instructions carefully.\n
            1. Return only a valid JSON that adheres to the format instructions.
            2. Do not add any additional text or explanation outside of the JSON object.
            3. Ensure the JSON is properly formatted and can be parsed without errors.
            Here are the format instructions:\n
            {format_instructions}
            """),
    ]
)

chain = prompt | llm
variables = {
    "text": parsed_args.text,
    "format_instructions": format_instructions
}

last_error_hint = "" #blank to start with
for attempt in range(parsed_args.max_retries + 1):
    if attempt == 0:
     
        #first attempt : vanilla invoke
        prompt_prepared = prompt.format(**variables)
    else:
        # retry with error hint appended to the user message
        retry_prompt = ChatPromptTemplate.from_messages(
            [ *prompt.messages,
                ("user",
                 f"""
                 The previous response had the following issue: {last_error_hint}
                 Please correct the response to adhere to the format instructions.
                 Here are the format instructions again:\n
                 {format_instructions}
                 """),
            ]
        )
        prompt_prepared = retry_prompt.format(**variables)
        
     # invoke the model
    response = llm.invoke(prompt_prepared)
    #print(f"Response from gpt-4o (attempt {attempt}):")  
    #print(response.content)
    content = response.content.strip()
    if content.startswith("```") and content.endswith("```"):
        # remove markdown code fences if present
        content = "\n".join(content.split("\n")[1:-1]).strip()
        
    try:
        _= json.loads(content)  # quick check for valid JSON
        
    except Exception as e_json:
        raise ValueError(f"Failed to parse JSON output: {e_json}")
    
    product : RestaurantInfo = parser.parse(content)
    print("\nParsed Product Info:")
    
    print(product.model_dump_json(indent=2))
    print("it was successfully parsed!")
    
    break  # exit the retry loop on success
        