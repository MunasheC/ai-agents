from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field

#======================>>>>>>>>>> Model setup
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=API_KEY)

#===================>>>>>>>>>>>>>> Configuring Agent

# Defining output structure
class BankingAnswer(BaseModel):
    term: str = Field(description="The banking term being explained")
    definition: str = Field(description="A simple explanation of the term")
    example: str = Field(description="A fun, practical example showing the term in use to best explain the term to a beginner")

parser = PydanticOutputParser(pydantic_object=BankingAnswer)
structured_model = model.with_structured_output(BankingAnswer)

#===========================>>>>>>>>>>> Agent interaction

def chat():
    print("Greetings! I am your interactive AI-powered banking dictionary. Ask me any banking term (type exit to quit) ")
    username = input("\nWhat can I call you?")
    if username:
        while True:
            term = input(f'{username}:')
            if term.lower() in ["exit", "quit"]:
                print("Thanks. Bye for now")
                break
            response = structured_model.invoke(term)
            print(f'\n Term: {response.term}')
            print(f'\n Definition: {response.definition}')
            print(f'\n Example: {response.example}')


chat()