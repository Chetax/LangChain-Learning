from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
load_dotenv()

prompt1=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8,google_api_key=os.environ["gemini_key"])

parser=StrOutputParser()

prompt2=PromptTemplate(
    template="Explain me the follwing joke {joke}",
    input_variables=['joke']
)

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)
print(chain.invoke({'topic':'AI'}))


