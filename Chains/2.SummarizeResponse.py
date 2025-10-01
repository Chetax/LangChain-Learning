from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()


prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Generate a 5 pointer summary from the follwing text\n {text}",
    input_variables=['text']
)

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8,google_api_key=os.environ["gemini_key"])

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model| parser

result=chain.invoke({'topic':'UnEmployment in India'})
print(result)
chain.get_graph().print_ascii()

