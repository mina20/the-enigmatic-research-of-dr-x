from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rouge_score import rouge_scorer
import sacrebleu
from sentence_transformers import SentenceTransformer, util


class FlanLLMHelper:
    def __init__(self, model_name="google/flan-t5-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def generate(self, input_text, max_new_tokens=256):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def evaluate(self, task, reference, generated, context=None):
        if task == "summary":
            rouge_scores = self.rouge.score(reference, generated)
            return {
                "rouge1": round(rouge_scores['rouge1'].fmeasure, 4),
                "rougeL": round(rouge_scores['rougeL'].fmeasure, 4)
            }
        
        elif task == "translation":
            bleu = sacrebleu.corpus_bleu([generated], [[reference]]).score
            return {
                "bleu": round(bleu, 2)
            }

        elif task == "qa":
            result = {}
            if context:
                emb_context = self.embedder.encode(context, convert_to_tensor=True)
                emb_generated = self.embedder.encode(generated, convert_to_tensor=True)
                context_score = float(util.pytorch_cos_sim(emb_context, emb_generated)[0])
                result["context_score"] = round(context_score, 4)

            if reference:
                result["exact_match"] = int(generated.strip().lower() == reference.strip().lower())

            return result

        else:
            return {"error": "Unsupported task"}