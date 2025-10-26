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

class ProductInfo(BaseModel):
    name: str = Field(..., description="The name of the product")
    category: str = Field(..., description="The category of the product, e.g., laptop, smartphone, headphones")
    price_estimate: Optional[float] = Field(None, description="Estimated price of the product in USD")
    
    pros : List[str] = Field(default_factory=List, description="List of pros of the product in bullet points")
    cons : List[str] = Field(default_factory=List, description="List of cons of the product in bullet points")
    
args = argparse.ArgumentParser(description="JSON Formatted Output with pydantic")
args.add_argument("--product_name", type=str, required=True, help="Name of the product")
args.add_argument("--description", type=str, required=True, help="Description of the product features and details")
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

parser = PydanticOutputParser(pydantic_object=ProductInfo)
format_instructions = parser.get_format_instructions()

print("Format Instructions:")
print(format_instructions)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """you are an information extraction assistant. extract clean product information from user input.
            if the field is unknown for the provided text, set it to null
            or an empty list (for collections), and never fabricate the facts.
            """),
        ("user",
            """
            Extract product information from the text and format it as per the instructions below:
            INPUT_NAME: {name}\n
            INPUT_Description: {description}\n
            Follow the format instructions carefully.\n
            1. Return only a valid JSON that adheres to the format instructions.
            2. Do not include any commentary or markdown fences.
            3. Adhere to JSON schema and field descriptions.          
            {format_instructions}
            """)        
    ]
)

chain = prompt | llm
variables = {
    "name": parsed_args.product_name,
    "description": parsed_args.description,
    "format_instructions": format_instructions
}

last_error_hint = "" #blank to start with
for attempt in range(parsed_args.max_retries + 1):
    if attempt == 0:
        #response = chain.invoke(variables)
        #print("Response from gpt-4o:")  
        #print(response.content)
        
        #first attempt : vanilla invoke
        prompt_prepared = prompt.format(**variables)
    else:
        # retry with error hint appended to the user message
        retry_prompt = ChatPromptTemplate.from_messages(
            [ *prompt.messages,
                ("user", 
                 f"""The previous output was invalid JSON.
                 here is the error encountered : {last_error_hint}. \n\n
                 please correct the output to adhere to the format instructions.
                 please return only the corrected JSON that follows the exact schema.print
                 dont include any extra text.""")
                ]       
        )
        prompt_prepared = retry_prompt.format(**variables)

    # invoke the model
    response = llm.invoke(prompt_prepared)
    print(f"Response from gpt-4o (attempt {attempt}):")  
    print(response.content)
    content = response.content.strip()
    if content.startswith("```") and content.endswith("```"):
        # remove markdown code fences if present
        content = "\n".join(content.split("\n")[1:-1]).strip()
        
    try:
        _= json.loads(content)  # quick check for valid JSON
        
    except Exception as e_json:
        raise ValueError(f"Failed to parse JSON output: {e_json}")
    
    product : ProductInfo = parser.parse(content)
    print("\nParsed Product Info:")
    
    print(product.model_dump_json(indent=2))
    print("it was successfully parsed!")
    
    break  # exit the retry loop on success
        