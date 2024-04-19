#Embeddings & use of ChomaDB vector DB  #chainlit is like streamlit but for LLM applications
# set HNSWLIB_NO_NATIVE=1
print("START Project #2, Vector DB")
import openai
import os
import chromadb

os.environ['OPENAI_API_KEY'] = 'sk-'
openai.api_key = 'sk-'

chroma_client = chromadb.Client()

#Collection is the Vector DB
collection = chroma_client.create_collection(name="my_collection")
collection.add (
    documents = ["precision farming", "yield enhancements"],  ##Tokenized split parts of a file
    metadatas = [{"source": "precision"}, {"source": "yield"}],  ## important for question answering system
    ids = ["id1", "id2"]  ##reach document will have unique ids
)

results = collection.query (
    query_texts = ["what to do for farming?"],
    n_results=2
)
print(results)
