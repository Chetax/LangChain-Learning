from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.schema.runnable import RunnableBranch ,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel ,Field 
from typing import Literal 
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",temperature=0.8,google_api_key=os.environ["gemini_key"])
parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
   template="Classify the Sentiment of the following feedback text into positive or negative:\n{feedback}\n{format_instructions}",
input_variables=['feedback'],
partial_variables={'format_instructions': parser2.get_format_instructions()}
)

prompt2=PromptTemplate(
    template='Write an approproiate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write an approproiate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain=prompt1 | model | parser2

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive' ,prompt2  | model |parser),
    (lambda x:x.sentiment=='negative' ,prompt3 | model |parser),
    RunnableLambda(lambda x: "Could not fing sentiment")
)

chain=classifier_chain | branch_chain

result=chain.invoke({'feedback':"This is a terrible phone"})

print(result)