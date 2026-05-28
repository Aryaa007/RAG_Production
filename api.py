from fastapi import FastAPI
from pydantic import BaseModel
from retriever import hybrid_retrieve
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import io


from fastapi.middleware.cors import CORSMiddleware


embed_model   = SentenceTransformer("BAAI/bge-small-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
active_collection = {"name": "dbms_notes"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React's URL
    allow_methods=["*"],
    allow_headers=["*"]
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG API running"
    }


# @app.post("/ask")
# def ask(q: Question):

#     results = hybrid_retrieve(
#         q.question
#     )

#     return {
#         "results": results
#     }

#connect to groq and chroma, pull chunks, re-rank, and return top results for a query
from answer import answer
#upload a file and ingest into chroma
from fastapi import UploadFile

def ingest_pdf(file_bytes, filename):
    collection_name = filename.replace(".pdf", "").replace(" ", "_").lower()

    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    embeddings = embed_model.encode(chunks)

    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    col = chroma_client.get_or_create_collection(name=collection_name)
    col.add(
        ids        = [f"chunk_{i}" for i in range(len(chunks))],
        documents  = chunks,
        embeddings = embeddings.tolist(),
        metadatas  = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    )

    return collection_name

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files supported"}

    content = await file.read()
    collection_name = ingest_pdf(content, file.filename)
    active_collection["name"] = collection_name

    return {
        "filename":   file.filename,
        "collection": collection_name,
        "message":    "File ingested successfully"
    }

@app.post("/ask")
def ask(q: Question):
    answer_text, chunks = answer(
        q.question,
        collection_name=active_collection["name"]
    )
    sources = list({chunk["source"] for chunk in chunks})
    return {"answer": answer_text, "sources": sources}

