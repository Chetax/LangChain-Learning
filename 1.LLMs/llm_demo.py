from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.environ["gemini_key"])

result=llm.invoke("Explain Langchain in 3 sentences.")
print(result.content)