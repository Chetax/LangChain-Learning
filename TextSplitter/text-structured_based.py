# \n\n -> represent Paragraph splitter 
# \n line 
# _ words based
# '' character

from langchain.text_splitter import RecursiveCharacterTextSplitter
text=""" 
LangChain provides a standard interface for working with vector stores, allowing users to easily switch between different vectorstore implementations.

The interface consists of basic methods for writing, deleting and searching for documents in the vector store.

The key methods are:
add_documents: Add a list of texts to the vector store.
delete: Delete a list of documents from the vector store.
similarity_search: Search for similar documents to a given query.
"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)