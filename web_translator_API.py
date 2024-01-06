from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    text: str
    
app = FastAPI()
model = pipeline('translation_ru_to_en', 'Helsinki-NLP/opus-mt-ru-en')

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    return model(item.text )[0]
