from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from utils import mask_pii, demask_pii
from models import load_model, classify_email

app = FastAPI()

class EmailRequest(BaseModel):
    email_body: str

class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

# Load model once at startup
model = None

@app.on_event("startup")
def load_classification_model():
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/classify_email", response_model=EmailResponse)
def classify_email_endpoint(request: EmailRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    original_email = request.email_body
    masked_email, masked_entities = mask_pii(original_email)
    category = classify_email(model, masked_email)

    response = {
        "input_email_body": original_email,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
    return response
