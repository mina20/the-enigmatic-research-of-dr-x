import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.utils import compute_rouge_scores
class QAEngine:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config["flan_model"]).to(self.device)
        self.top_k = config["top_k"]
        self.max_length = config["max_length"]
        

    def generate_answer(self, question, context):
        prompt = (
        "You are a knowledgeable assistant. Use only the information in the context to answer the question. "
        # "If the context doesn't provide a clear answer, respond with: \"I don’t have enough information.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        outputs = self.model.generate(inputs["input_ids"], max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def ask_question(self, question, chunks, chunk_embeddings, top_k, embed_fn):
        question_embedding = embed_fn([question])[0]
        scores = [torch.cosine_similarity(torch.tensor(chunk_emb), torch.tensor(question_embedding), dim=0).item()
                  for chunk_emb in chunk_embeddings]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_chunks = [chunks[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        context = "\n\n".join([chunk["chunk_text"] for chunk in top_chunks])
        answer = self.generate_answer(question, context)

        rouge_scores = compute_rouge_scores(question, context, answer)
        # rougeL_q_ans = compute_rouge_scores(question, answer)
        # rougeL_ctx_ans = compute_rouge_scores(context, answer)

        return {
            "question": question,
            "answer": answer,
            "top_chunks": top_chunks,
            "similarities": top_scores,
            "scores": rouge_scores,
            # "rougeL_q_ans": rougeL_q_ans,
            # "rougeL_ctx_ans": rougeL_ctx_ans
        }