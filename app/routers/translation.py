from fastapi import APIRouter
from pydantic import BaseModel
from translation_engine import Translator
from utils.utils import compute_bleu  

router = APIRouter()

# ========== Request ==========
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    reference: str = ""  

# ========== Endpoint ==========
@router.post("/translate")
def translate_text(request: TranslationRequest):
    translator = Translator(src_lang=request.source_lang, tgt_lang=request.target_lang)
    
    translated = translator.translate(request.text)

    score = None
    if request.reference.strip():
        score = compute_bleu(reference=request.reference, prediction=translated)

    return {
        "translated_text": translated,
        "score": score
    }