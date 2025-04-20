import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.utils import compute_summary_rouge

class SummaryEngine:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config["flan_model"]).to(self.device)
        self.max_length = config["max_length"]

    def generate_summary(self, context):
        prompt = (
            "You are a helpful assistant. Summarize the following content concisely:\n\n"
            f"{context}\n\nSummary:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        outputs = self.model.generate(inputs["input_ids"], max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, context):
        summary = self.generate_summary(context)
        rouge_scores = compute_summary_rouge(reference=context, summary=summary)

        return {
            "context": context,
            "summary": summary,
            "rouge": rouge_scores
        }