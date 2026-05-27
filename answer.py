from groq import Groq
from retriever import hybrid_retrieve

# ── setup ──────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)

# ── build prompt ───────────────────────────────────────────────────────────────
# This is the most important part.
# We inject the retrieved chunks into the prompt as "context"
# and tell the LLM to ONLY answer from that context.
# This is what prevents hallucination.

def build_prompt(question, chunks):

    context = ""

    for i, chunk in enumerate(chunks, start=1):
        context += f"[{i}] Source: {chunk['source']}\n"
        context += f"{chunk['text']}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.

Rules:
- After every claim, cite the source like this: [Source: filename]
- If the context does not contain enough information, say "I don't have enough information to answer this."
- Do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

    return prompt

# ── generate answer ────────────────────────────────────────────────────────────

def answer(question, collection_name="dbms_notes"):
    chunks = hybrid_retrieve(question, collection_name)

    if not chunks:
        print("No relevant chunks found.")
        return

    # Step 2 — build the prompt with chunks injected
    prompt = build_prompt(question, chunks)

    # Step 3 — send to Groq (Llama 3.3 70B)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Step 4 — extract the answer text
    answer_text = response.choices[0].message.content

    return answer_text, chunks

# ── run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RAG Question Answering — type 'exit' to quit\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        if not question:
            continue

        print("\nRetrieving and generating answer...\n")

        result, chunks = answer(question)

        print("=" * 60)
        print("Answer:\n")
        print(result)
        print("\n" + "=" * 60)
        print("Retrieved from:")
        for i, c in enumerate(chunks, start=1):
            print(f"  [{i}] {c['source']}")
        print()