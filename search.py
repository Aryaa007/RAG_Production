import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="dbms_notes"
)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

query = input("Ask a question: ")

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=3
)

for i, doc in enumerate(results["documents"][0], start=1):
    print(f"\n--- Result {i} ---")
    print(doc[:500])