from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8,google_api_key=os.environ["gemini_key"])

prompt1=PromptTemplate(
    template='Write a joke and assumption about {topic} which is false about the topic',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Write a core truth about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()
parallel_Chain=RunnableParallel({
    'joke':RunnableSequence(prompt1,model,parser),
    'truth':RunnableSequence(prompt2,model,parser)
}
)

result=parallel_Chain.invoke({'topic':'friendship'})
print(result['joke'])
print(result['truth'])