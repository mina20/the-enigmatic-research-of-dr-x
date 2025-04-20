from fastapi import FastAPI
from routers import qa, summary, translation 
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
app = FastAPI()

app.include_router(qa.router, prefix="/qa")
app.include_router(summary.router, prefix="/summary")
app.include_router(translation.router, prefix="/translate")