# Import from the more stable langchain_community package
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# This is the standard environment variable name the library looks for
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face API token not found. Make sure it's in your .env file.")

# This code now uses the HuggingFaceEndpoint from langchain-community,
# which should be compatible with your downgraded huggingface-hub.
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=100
)

result = llm.invoke("What is the capital of India")

print(result)
