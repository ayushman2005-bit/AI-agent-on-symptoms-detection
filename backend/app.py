from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Health AI Agent API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
# In a production environment, you'd load the fine-tuned model
# For this template, we'll use a placeholder or logic to load the model
MODEL_PATH = "./saved_model"
DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

try:
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        # Fallback to base model for demonstration if not trained
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL, num_labels=41) # 41 is the number of diseases in the dataset
    
    le = joblib.load('label_encoder.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/predict")
async def predict(input_data: SymptomInput):
    if not model or not tokenizer:
        # Mock prediction if model isn't loaded for some reason
        return {"disease": "Sample Disease (Model not loaded)", "confidence": 0.95}
    
    inputs = tokenizer(input_data.symptoms, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
    disease = le.inverse_transform([pred.item()])[0]
    
    return {
        "disease": disease,
        "confidence": float(conf.item())
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
