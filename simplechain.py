from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key=os.getenv("OPEN_AI_API_KEY")


model=ChatOpenAI(model="gpt-4o-mini",
                 openai_api_key=api_key)

prompt=PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

chain= prompt|model|parser
result=chain.invoke({'topic':'cricket'})
print(result)
chain.get_graph().print_ascii()
