import chromadb
from sentence_transformers import SentenceTransformer


chunks = [
    "Advantages of DBMS over File System",
    "Entity Relationship Model",
    "Normalization in DBMS"
]

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

embeddings = model.encode(chunks)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="dbms_notes"
)

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("Stored successfully!")