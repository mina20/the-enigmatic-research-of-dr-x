from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import csv
from typing import List, Optional
from qa_engine import QAEngine
from text_extractor import FileTextExtractor
from chunk_embedder import ChunkEmbedder
from vector_db import SimpleVectorDB

app = FastAPI()

config = None
chunks = None
embeddings = None
chunker = None
qa_model = None
id_counter = 1
# ========== Data Models ==========
class QARequest(BaseModel):
    question: str

# ========== Startup Initialization ==========
@app.on_event("startup")
def startup_event():
    global config, chunks, embeddings, chunker, qa_model, id_counter

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    extractor = FileTextExtractor(config["data_directory"])
    all_text = extractor.extract_all()

    chunker = ChunkEmbedder(config)
    chunks, embeddings = chunker.chunk_and_embed(all_text)
    print(f"Extracted {len(chunks)} chunks.")

    vectordb = SimpleVectorDB(embedding_dim=embeddings.shape[1])
    vectordb.add(embeddings, chunks)

    qa_model = QAEngine(config=config)

# ========== Endpoint ==========
@app.post("/ask")
def ask_question(request: QARequest):
    global id_counter

    question = request.question.strip()
    if not question:
        return {"error": "Question cannot be empty."}

    out = qa_model.ask_question(
        question,
        chunks,
        embeddings,
        top_k=config.get("retriever_top_k", 5),
        embed_fn=chunker.embed_chunks
    )

    # save results to CSV
    output_file = config.get("output_csv", "qa_output.csv")
    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "question", "answer", "rougeL_q_ctx", "rougeL_q_ans", "rougeL_ctx_ans"
        ])
        if id_counter == 1:
            writer.writeheader()
        writer.writerow({
            "id": id_counter,
            "question": out["question"],
            "answer": out["answer"],
            # "scores": out["rouge_scores"]
            "rougeL_q_ctx": out["scores"]["rougeL_q_ctx"],
            "rougeL_q_ans": out["scores"]["rougeL_q_ans"],
            "rougeL_ctx_ans": out["scores"]["rougeL_ctx_ans"]
        })

    id_counter += 1
    # return out
    return {
        "question": out["question"],
        "answer": out["answer"],
        "retrieved_chunks": [
            {"chunk": c, "similarity": s}
            for c, s in zip(out["top_chunks"], out["similarities"])
        ],
        "rouge": {
            "q_ctx": out["scores"]["rougeL_q_ctx"],
            "q_ans": out["scores"]["rougeL_q_ans"],
            "ctx_ans": out["scores"]["rougeL_ctx_ans"]
        }
    }