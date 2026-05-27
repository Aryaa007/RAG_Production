
from sentence_transformers import SentenceTransformer
import chromadb

COLLECTION  = "dbms_notes"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K       = 5   # how many chunks to retrieve


model      = SentenceTransformer(EMBED_MODEL)
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
  
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings = query_embedding.tolist(),
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text":        doc,
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "score":       round(1 - dist, 4)   # cosine similarity (higher = better)
        })

    return retrieved


def display_results(query: str, results: list[dict]):
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] Source: {r['source']}  |  Chunk #{r['chunk_index']}  |  Score: {r['score']}")
        print("-" * 60)
        print(r["text"][:400].strip())
        print("...")



if __name__ == "__main__":
    print("RAG Query Interface — type 'exit' to quit\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        if not query:
            continue

        results = retrieve(query)
        display_results(query, results)

        # Show which documents were cited
        sources = list({r["source"] for r in results})
        print(f"\nCited from: {', '.join(sources)}")
        print()