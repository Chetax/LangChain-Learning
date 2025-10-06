from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
text=""" 
    from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

prompt=PromptTemplate(
    template="Generate 5 interesting facts about {topic} ,with each fact about 5-10 words",
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8,google_api_key=os.environ["gemini_key"])

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({'topic':'cricket'})

print(result)
chain.get_graph().print_ascii()

"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)