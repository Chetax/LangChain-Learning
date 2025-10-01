from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.8,google_api_key=os.environ["gemini_key"])

result=llm.invoke("Give me 5 indian name of man")
print(result.content)