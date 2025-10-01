import os
from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import GoogleGenerativeAIEmbeddings

api_key = os.environ.get("gemini_key")
if not api_key:
    raise ValueError("API key not found. Make sure 'gemini_key' is set in your .env file.")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    google_api_key=api_key,
    dimensions=32
)



document=[
    "Delhi"
]
result=embeddings.embed_documents(document)
print(len(result[0]))