import json
import time
from pathlib import Path
from src.search import RAGSearch

project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "data" / "evaluation_dataset.json"
output_path = project_root / "data" / "rag_outputs.json"

with open(dataset_path, encoding="utf-8") as f:
    dataset = json.load(f)

rag = RAGSearch()
results = []

for item in dataset:
    question = item["question"]
    start = time.time()
    query_embedding = rag.embedding_pipeline.model.encode(question)
    retrieved_chunks = rag.vectorstore.query(query_embedding, top_k=3)
    answer = rag.search_and_summarize(question)
    end = time.time()

    latency = end - start

    contexts = [chunk["text"] for chunk in retrieved_chunks]
    sources = [
        {
            "source": chunk["metadata"].get("source"),
            "page": chunk["metadata"].get("page")
        }
        for chunk in retrieved_chunks
    ]

    results.append({
        "question": question,
        "expected_source": item["source_pdf_name"],
        "expected_page": item["expected_page_number"],
        "model_answer": answer,
        "retrieved_contexts": contexts,
        "retrieved_sources": sources,
        "latency": latency
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Evaluation run completed.")