import chromadb
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.abspath(os.path.curdir)
DEFAULT_PATH = os.path.join(os.path.abspath(os.path.join(BASE_DIR,"..")),"app","db")

client = chromadb.PersistentClient(
    path = DEFAULT_PATH
)

collection = client.get_collection("pdf_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrievar(query, k=3):
    query_embedding = model.encode(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    return [
        {
            "text":doc,
            "metadata":meta
        } 
        for doc, meta in zip(documents, metadatas)
    ]

query = "What is fast api ?"
results = retrievar(query=query)
for r in results:
    print("\n---")
    print(r["text"])
    print(r["metadata"])
    


