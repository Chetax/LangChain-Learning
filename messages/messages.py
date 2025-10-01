from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8,google_api_key=os.environ["gemini_key"])

message=[
    SystemMessage(content='Your are a helpful assistant'),
    HumanMessage(content="Tell me about consultadd company")
]

result=model.invoke(message)
message.append(AIMessage(content=result.content))

print(message)