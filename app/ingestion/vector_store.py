import chromadb
from sentence_transformers import SentenceTransformer
from loader import load_pdfs
import os

BASE_DIR = os.path.abspath(os.path.curdir)
DEFAULT_PATH = os.path.join(os.path.abspath(os.path.join(BASE_DIR,"..")),'app',"db")

client = chromadb.PersistentClient(
    path="./db"
)

collection = client.get_or_create_collection(name="pdf_docs")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_pdfs(max_pages=3000)

texts = [doc["text"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]

embeddings = embedding_model.encode(texts)

collection.add(
    documents=texts,
    metadatas=metadatas,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(texts))]
)
