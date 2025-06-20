from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key=os.getenv("OPEN_AI_API_KEY")


prompt1=PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='write the summary of the {text}\n',
    input_variables=['text']
)

model=ChatOpenAI(model="gpt-4o-mini",
                 openai_api_key=api_key)

parser=StrOutputParser()

chain=prompt1|model|parser|prompt2|model|parser

result=chain.invoke({
    'topic':'pakistan economy'
})

print(result)
chain.get_graph().print_ascii()
