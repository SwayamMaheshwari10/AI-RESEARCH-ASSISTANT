import json
import os
import numpy as np
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy,
)

load_dotenv()

project_root = Path(__file__).resolve().parent.parent
evaluation_dataset_path = project_root / "data" / "evaluation_dataset.json"
rag_outputs_path = project_root / "data" / "rag_outputs.json"

# ------------------------
# Load evaluation dataset
# ------------------------

with open(evaluation_dataset_path, encoding="utf-8") as f:
    evaluation_data = json.load(f)

ground_truth_map = {
    item["question"]: item["ground_truth_answer"]
    for item in evaluation_data
}

# ------------------------
# Load RAG outputs
# ------------------------

with open(rag_outputs_path, encoding="utf-8") as f:
    rag_outputs = json.load(f)

questions = []
answers = []
contexts = []
ground_truths = []
latencies = []

for item in rag_outputs:

    questions.append(item["question"])
    answers.append(item["model_answer"])
    contexts.append(item["retrieved_contexts"])
    ground_truths.append(ground_truth_map[item["question"]])
    latencies.append(item["latency"])

# ------------------------
# Convert to HF dataset
# ------------------------

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# ------------------------
# Initialize models
# ------------------------

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Reduce hallucination sensitivity
answer_relevancy.strictness = 1

# ------------------------
# Run RAGAS evaluation
# ------------------------

ragas_results = evaluate(
    dataset,
    metrics=[
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy
    ],
    llm=llm,
    embeddings=embeddings,
    run_config=RunConfig(
        max_workers=1,
        max_retries=3,
        timeout=120
    ),
    batch_size=2
)

# ------------------------
# Compute metrics
# ------------------------

avg_latency = np.mean(latencies)

context_recall_score = np.nanmean(ragas_results["context_recall"])
context_precision_score = np.nanmean(ragas_results["context_precision"])
faithfulness_score = np.nanmean(ragas_results["faithfulness"])
answer_relevancy_score = np.nanmean(ragas_results["answer_relevancy"])

# ------------------------
# Print results
# ------------------------

print("\n==============================")
print("RAG SYSTEM EVALUATION RESULTS")
print("==============================\n")

print(f"Total Queries Evaluated: {len(questions)}")

print("\nSystem Performance")
print("------------------")
print(f"Average Latency: {avg_latency:.3f} seconds")

print("\nRetriever Evaluation")
print("--------------------")
print(f"Context Recall: {context_recall_score:.3f}")
print(f"Context Precision: {context_precision_score:.3f}")

print("\nGenerator Evaluation")
print("--------------------")
print(f"Faithfulness: {faithfulness_score:.3f}")
print(f"Answer Relevancy: {answer_relevancy_score:.3f}")

print("\nEvaluation completed.\n")