import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize ChatOpenAI with API key from environment
# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", 
#     temperature=0.7,
#     openai_api_key=os.getenv("OPENAI_API_KEY")
# )
# print(llm.invoke("Hello Lanchain!"))  # type: ignore

model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}.")

chain = prompt | model

response = chain.invoke({"topic": "water"})
print(response.content)  # type: ignore


