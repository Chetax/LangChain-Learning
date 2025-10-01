"""
Embedding help you find the similiarites between the other sentences and given query .
Example if we have 5 sentence ,each sentence telling something about the person
Example : 

documents = [
"Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
"Ms Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
"Sachin Tendulkar, also known as the "God of Cricket', holds many batting records.",
"Rohit Sharma is known for his elegant batting and record-breaking double centuries.
"Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]
query ="tell me about virat kohli"

here we willl get the value for each sentnece  which sugeest similarity or sementatic simalrites b/w other sentences
"""


import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings


api_key = os.environ.get("gemini_key")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    google_api_key=api_key
)

documents = [
"Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
"Ms Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
"Sachin Tendulkar, also known as the God of Cricket, holds many batting records.",
"Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
"Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query="Tell me about Ms Dhoni "

doc_embedding=embeddings.embed_documents(documents)
query_embedding=embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)
scores_list = scores[0]  

index, score = sorted(list(enumerate(scores_list)), key=lambda x: x[1])[-1]

print(documents[index])
print("Similarity score is:", score)
