import yaml
import csv
from qa_engine import QAEngine
from utils import compute_rouge_scores
from text_extractor import FileTextExtractor
from chunk_embedder import ChunkEmbedder
from vector_db import SimpleVectorDB


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    directory = config["data_directory"]
    extractor = FileTextExtractor(directory)
    all_text = extractor.extract_all()
    # Chunking and embedding
    chunker = ChunkEmbedder(config)
    chunks, embeddings = chunker.chunk_and_embed(all_text)
    print(f"Extracted {len(chunks)} chunks from {len(all_text)} files.")

    # Set up Vector DB
    vectordb = SimpleVectorDB(embedding_dim=embeddings.shape[1])
    vectordb.add(embeddings, chunks)

    # Initialize QAEngine with vector DB and embedder
    qa_model = QAEngine(config=config)

    print("\nEnter questions (type 'exit' to quit):\n")
    id_counter = 1

    while True:
        question = input("❓ Question: ").strip()
        if question.lower() == "exit":
            break
        out = qa_model.ask_question(question, chunks, embeddings, top_k=5, embed_fn=chunker.embed_chunks)

        print(f"\nQ: {out['question']}")
        print(f"A: {out['answer']}")
        for i, (chunk, sim) in enumerate(zip(out["top_chunks"], out["similarities"]), 1):
            print(f"   Chunk {i} (Sim: {sim:.4f}): {chunk[:100]}...")

        print(f"ROUGE-L — Q↔Ctx: {out['rougeL_q_ctx']:.4f}, Q↔Ans: {out['rougeL_q_ans']:.4f}, Ctx↔Ans: {out['rougeL_ctx_ans']:.4f}")
        print("=" * 100)

        # Save results to CSV
        with open(config.get("output_csv", "qa_output.csv"), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "question", "answer", "rougeL_q_ctx", "rougeL_q_ans", "rougeL_ctx_ans"
            ])
            if id_counter == 1:
                writer.writeheader()
            writer.writerow({
                "id": id_counter,
                "question": out["question"],
                "answer": out["answer"],
                "rougeL_q_ctx": out["rougeL_q_ctx"],
                "rougeL_q_ans": out["rougeL_q_ans"],
                "rougeL_ctx_ans": out["rougeL_ctx_ans"]
            })
        id_counter += 1


if __name__ == "__main__":
    main()