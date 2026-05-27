
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


DOCS_FOLDER   = "Docs"
COLLECTION    = "dbms_notes"
CHROMA_PATH   = "./chroma_db"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 800  
CHUNK_OVERLAP = 100


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "] 
)

all_chunks    = []   
all_metadatas = []   

for filename in os.listdir(DOCS_FOLDER):
    if not filename.endswith(".pdf"):
        continue

    pdf_path  = os.path.join(DOCS_FOLDER, filename)
    file_text = ""

    print(f"\nReading: {filename}")

    reader = PdfReader(pdf_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            file_text += text + "\n"

    # Chunk this file's text separately so chunks never span two documents
    file_chunks = splitter.split_text(file_text)

    print(f"  → {len(reader.pages)} pages | {len(file_text):,} chars | {len(file_chunks)} chunks")

    for i, chunk in enumerate(file_chunks):
        all_chunks.append(chunk)
        all_metadatas.append({
            "source":      filename,        # which PDF this came from
            "chunk_index": i,               # position within that file
            "chunk_total": len(file_chunks) # total chunks in that file
        })

print(f"\nTotal chunks across all files: {len(all_chunks)}")




model      = SentenceTransformer(EMBED_MODEL)
embeddings = model.encode(all_chunks, show_progress_bar=True)

print(f"\nEmbeddings shape: {len(embeddings)} x {len(embeddings[0])}")

#Store in ChromaDB

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Always start fresh on re-ingest
try:
    client.delete_collection(COLLECTION)
    print(f"Deleted existing collection: {COLLECTION}")
except Exception:
    pass

collection = client.get_or_create_collection(name=COLLECTION)

collection.add(
    ids        = [f"chunk_{i}" for i in range(len(all_chunks))],
    documents  = all_chunks,
    embeddings = embeddings.tolist(),
    metadatas  = all_metadatas   # source + chunk_index stored per chunk
)

print(f"Stored {len(all_chunks)} chunks in '{COLLECTION}' successfully!")

# Quick Sanity Check 

print("\n" + "=" * 50)
print("STEP 4: Sanity Check — Test Query")
print("=" * 50)

TEST_QUERY = "What is normalization in DBMS?"

query_embedding = model.encode([TEST_QUERY])

results = collection.query(
    query_embeddings = query_embedding.tolist(),
    n_results        = 3,
    include          = ["documents", "metadatas", "distances"]
)

print(f"\nQuery: '{TEST_QUERY}'\n")

for rank, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
), start=1):
    print(f"Result #{rank}")
    print(f"  Source : {meta['source']}  (chunk {meta['chunk_index']} of {meta['chunk_total']})")
    print(f"  Score  : {1 - dist:.4f}")   # convert distance → similarity
    print(f"  Preview: {doc[:200].strip()}...")
    print()

print("Ingestion complete. Ready for query.py ✓")