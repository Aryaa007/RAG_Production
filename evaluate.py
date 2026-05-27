import json
from groq import Groq
from retriever import hybrid_retrieve

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)

with open("golden_dataset.json", "r") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions from golden dataset\n")

#generate RAG answer

def get_rag_answer(question):

    chunks = hybrid_retrieve(question)

    if not chunks:
        return "No relevant chunks found."

    context = ""
    for i, chunk in enumerate(chunks, start=1):
        context += f"[{i}] Source: {chunk['source']}\n{chunk['text']}\n\n"

    prompt = f"""Answer the question using ONLY the context below.
Be concise. If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# ── judge answer quality ───────────────────────────────────────────────────────
# We use Groq itself as a judge.
# We show it: the question, the correct answer, the RAG answer
# and ask it to give a score from 0 to 1.
# This is called "LLM as a judge" — used by real AI teams.

def judge_answer(question, correct_answer, rag_answer):

    prompt = f"""You are an evaluation judge. Score how well the RAG answer matches the correct answer.

Question: {question}
Correct Answer: {correct_answer}
RAG Answer: {rag_answer}

Rules:
- Score 1.0 if the RAG answer is fully correct and covers all key points
- Score 0.5 if the RAG answer is partially correct
- Score 0.0 if the RAG answer is wrong or says "I don't know"

Reply with ONLY a number: 0.0, 0.5, or 1.0"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    score_text = response.choices[0].message.content.strip()

    # safely parse the score
    try:
        score = float(score_text)
    except:
        score = 0.0

    return score


results = []

for i, item in enumerate(dataset, start=1):
    question       = item["question"]
    correct_answer = item["answer"]

    print(f"[{i}/{len(dataset)}] {question}")

    # get RAG answer
    rag_answer = get_rag_answer(question)

    # judge it
    score = judge_answer(question, correct_answer, rag_answer)

    results.append({
        "question":       question,
        "correct_answer": correct_answer,
        "rag_answer":     rag_answer,
        "score":          score
    })

    print(f"  Score: {score}")
    print()

#final report

total_score  = sum(r["score"] for r in results)
average      = total_score / len(results)
passed       = sum(1 for r in results if r["score"] >= 0.5)
failed       = len(results) - passed
THRESHOLD    = 0.7  

print("=" * 60)
print("EVALUATION REPORT")
print("=" * 60)
print(f"Total questions : {len(results)}")
print(f"Passed (>=0.5)  : {passed}")
print(f"Failed (<0.5)   : {failed}")
print(f"Average score   : {average:.2f}")
print(f"Threshold       : {THRESHOLD}")
print()

if average >= THRESHOLD:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL — quality dropped below threshold")


with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDetailed results saved to eval_results.json")


import sys
if average < THRESHOLD:
    sys.exit(1)