from fastapi import APIRouter
from pydantic import BaseModel
import yaml
import csv
import torch
from typing import List

from summary_engine import SummaryEngine
from services.text_extractor import FileTextExtractor
from services.chunk_embedder import ChunkEmbedder
from services.vector_db import SimpleVectorDB
from utils.utils import compute_summary_rouge  

router = APIRouter()

config = None
chunks = None
embeddings = None
chunker = None
summary_model = None
id_counter = 1

# ========== Data Model ==========
class SummaryRequest(BaseModel):
    query: str

# ========== Startup ==========
@router.on_event("startup")
def startup_event():
    global config, chunks, embeddings, chunker, summary_model, id_counter

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    extractor = FileTextExtractor(config["data_directory"])
    all_text = extractor.extract_all()

    chunker = ChunkEmbedder(config)
    chunks, embeddings = chunker.chunk_and_embed(all_text)
    print(f"Extracted {len(chunks)} chunks.")

    vectordb = SimpleVectorDB(embedding_dim=embeddings.shape[1])
    vectordb.add(embeddings, chunks)

    summary_model = SummaryEngine(config=config)

# ========== Summary Endpoint ==========
@router.post("/summary")
def generate_summary(request: SummaryRequest):
    global id_counter

    query = request.query.strip()
    if not query:
        return {"error": "Query cannot be empty."}

    query_embedding = chunker.embed_chunks([query])[0]

    scores = [
        torch.cosine_similarity(torch.tensor(chunk_emb), torch.tensor(query_embedding), dim=0).item()
        for chunk_emb in embeddings
    ]

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    top_chunks = [chunks[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    context = "\n\n".join([chunk["chunk_text"] for chunk in top_chunks])
    summary = summary_model.summarize(context)
    rouge_score = compute_summary_rouge(reference=context, summary=summary["summary"])

    # Save the summary and scores to a CSV file
    output_file = config.get("output_csv", "summary_output.csv")
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "query", "summary", "rougeL"])
        if id_counter == 1:
            writer.writeheader()
        writer.writerow({
            "id": id_counter,
            "query": query,
            "summary": summary,
            "rougeL": rouge_score
        })

    id_counter += 1
    return {
        "query": query,
        "summary": summary,
        "top_chunks": [
            {"chunk": chunks[i], "similarity": top_scores[j]}
            for j, i in enumerate(top_indices)
        ],
        "rouge": {"rougeL": rouge_score}
    }