from rank_bm25 import BM25Okapi
import chromadb
import re


#pulling chunks from chromadb
COLLECTION = "dbms_notes"
CHROMA_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

print("Loading chunks...")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION)


raw = collection.get(
    include=["documents", "metadatas"]
)

all_chunks = raw["documents"]
all_metadatas = raw["metadatas"] 


print(f"Loaded {len(all_chunks)} chunks")

#tokenization for lower, cleaning and splitting into words
def tokenize(text):
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text.split()

#building the BM25 index
tokenized_corpus = [
    tokenize(chunk)
    for chunk in all_chunks
]

bm25 = BM25Okapi(
    tokenized_corpus
)

print("BM25 index built")

# add this after the existing bm25 index build at top
current_bm25_collection = {"name": "dbms_notes"}

def get_bm25(collection_name):
    global bm25, all_chunks, all_metadatas

    # only rebuild if collection changed
    if current_bm25_collection["name"] == collection_name:
        return bm25, all_chunks, all_metadatas

    print(f"Rebuilding BM25 for collection: {collection_name}")

    col = chroma_client.get_collection(name=collection_name)
    raw = col.get(include=["documents", "metadatas"])

    all_chunks    = raw["documents"]
    all_metadatas = raw["metadatas"]

    tokenized = [tokenize(c) for c in all_chunks]
    bm25      = BM25Okapi(tokenized)

    current_bm25_collection["name"] = collection_name

    return bm25, all_chunks, all_metadatas

#search function using BM25 to retrieve relevant chunks based on query tokens and returning their scores for relevance ranking 
import numpy as np

def bm25_search(query, collection_name="dbms_notes", top_k=5):
    index, chunks, metadatas = get_bm25(collection_name)

    query_tokens = tokenize(query)
    scores       = index.get_scores(query_tokens)
    top_indices  = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "id":     f"chunk_{idx}",
            "text":   chunks[idx],
            "source": metadatas[idx]["source"],
            "score":  float(scores[idx]),
            "rank":   rank
        })
    return results

#semantic search function

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def semantic_search(query, collection_name="dbms_notes", top_k=5):
    col = chroma_client.get_collection(name=collection_name)
    query_embedding = embed_model.encode([query])
    results = col.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    output = []

    for rank, (
        doc,
        meta,
        dist
    ) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        output.append({
            "id": f"chunk_{meta['chunk_index']}",
            "text": doc,
             "source": meta["source"],
            "rank": rank,
            "score": 1 - dist
        })

    return output

#RRF function to combine BM25 and semantic search results based on their ranks and scores to produce a final ranked list of relevant chunks for a given query
def reciprocal_rank_fusion(
    bm25_results,
    semantic_results,
    k=60
):

    scores = {}
    chunk_data = {}

    for result in bm25_results:

        cid = result["id"]

        if cid not in scores:
            scores[cid] = 0.0
            chunk_data[cid] = result

        scores[cid] += 1 / (k + result["rank"])

    for result in semantic_results:

        cid = result["id"]

        if cid not in scores:
            scores[cid] = 0.0
            chunk_data[cid] = result

        scores[cid] += 1 / (k + result["rank"])

    ranked_ids = sorted(
        scores,
        key=lambda x: scores[x],
        reverse=True
    )

    merged = []

    for rank, cid in enumerate(
        ranked_ids,
        start=1
    ):

        item = chunk_data[cid].copy()

        item["rrf_score"] = scores[cid]
        item["rank"] = rank

        merged.append(item)

    return merged

#test
# query = "What is normalization in DBMS?"

# bm25_results = bm25_search(query)

# semantic_results = semantic_search(query)

# merged = reciprocal_rank_fusion(
#     bm25_results,
#     semantic_results
# )

# for item in merged[:5]:

#     print(
#         item["id"],
#         round(item["rrf_score"], 6)
#     )

#cross encoder re-ranking function to further refine the relevance of retrieved chunks by using a cross-encoder model to compute similarity scores between the query and each retrieved chunk, allowing for a more accurate ranking of results based on their contextual relevance to the query.

from sentence_transformers import CrossEncoder

rerank_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded")

def rerank(
    query,
    candidates,
    top_k=5
):

    pairs = [
        (query, item["text"])
        for item in candidates
    ]

    scores = rerank_model.predict(
        pairs
    )

    for i, item in enumerate(candidates):
        item["rerank_score"] = float(
            scores[i]
        )

    ranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]

#test
# query = "What is normalization in DBMS?"

# bm25_results = bm25_search(query)

# semantic_results = semantic_search(query)

# merged = reciprocal_rank_fusion(
#     bm25_results,
#     semantic_results
# )

# reranked = rerank(
#     query,
#     merged[:10]
# )

# for item in reranked:

#     print(
#         item["rerank_score"]
#     )

#     print(
#         item["text"][:250]
#     )

#     print("-" * 50)

#Refactor in a single function to retrieve relevant chunks for a query by combining BM25 and semantic search results using reciprocal rank fusion, followed by cross-encoder re-ranking to produce a final ranked list of relevant chunks for the given query.
def hybrid_retrieve(query, collection_name="dbms_notes"):
    bm25_results     = bm25_search(query, collection_name)  # ← add collection_name
    semantic_results = semantic_search(query, collection_name)
    merged           = reciprocal_rank_fusion(bm25_results, semantic_results)
    final            = rerank(query, merged[:10])
    return final

#test the full retrieval pipeline with a sample query and print the top results along with their sources and re-rank scores to verify the effectiveness of the hybrid retrieval approach in fetching relevant chunks from the stored documents based on the input query.
if __name__ == "__main__":
    test_query = "What is normalization in DBMS?"
 
    print(f"Query: '{test_query}'")
    print("Running hybrid retrieval...\n")
 
    results = hybrid_retrieve(test_query)
 
    for i, r in enumerate(results, start=1):
       print(f"[{i}] {r['source']}  |  Re-rank score: {r['rerank_score']:.4f}")
       print(f"     {r['text'][:300].strip()}...")
       print()
 
    print(f"\nTop sources: {list({r['source'] for r in results})}")
 