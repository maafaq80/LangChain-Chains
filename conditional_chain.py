## if potive review then give response positve otherwise negative 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

# from langchain_core.runnables import RunnableParallel

load_dotenv()

api_key=os.getenv("OPEN_AI_API_KEY")

model=ChatOpenAI(model="gpt-4o-mini",
                 openai_api_key=api_key)

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="Tell about the sentiment of the review")
        
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template='Classify the sentiment of the text in positive or negative \n {feedback}\n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

prompt2=PromptTemplate(
    template="write an appropriate response for the positive feedback \n{feedback}",
    input_variables=['feedback']
    
)

prompt3=PromptTemplate(
    template="write an appropriate response for the negative feedback \n{feedback}",
    input_variables=['feedback']
    
)

classifier_chain=prompt1|model|parser2

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive' , prompt2|model|parser),
    (lambda x:x.sentiment=='negative' , prompt3|model|parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain=classifier_chain|branch_chain
result=chain.invoke(
   { 'feedback':'this is a beautiful phone'}
)

print(result)

chain.get_graph().print_ascii()




