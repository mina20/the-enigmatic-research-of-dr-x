from transformers import MarianMTModel, MarianTokenizer
import torch

class Translator:
    def __init__(self, src_lang: str = "en", tgt_lang: str = "fr"):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def translate(self, text: str) -> str:
        tokens = self.tokenizer.prepare_seq2seq_batch([text], return_tensors="pt").to(self.device)
        translated = self.model.generate(**tokens)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)