from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Health AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths relative to THIS file's directory (backend/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.joblib")
DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

model = None
tokenizer = None
le = None

def load_model():
    global model, tokenizer, le
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading fine-tuned model from {MODEL_PATH}...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        else:
            print(f"No saved model found at {MODEL_PATH}. Loading base BioBERT model...")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL, num_labels=41)

        if os.path.exists(LABEL_ENCODER_PATH):
            le = joblib.load(LABEL_ENCODER_PATH)
            print("Label encoder loaded successfully.")
        else:
            print(f"WARNING: label_encoder.joblib not found at {LABEL_ENCODER_PATH}.")
            le = None

        model.eval()
        print("Model loaded and ready.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None
        tokenizer = None
        le = None

# Load on startup
load_model()

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/predict")
async def predict(input_data: SymptomInput):
    symptoms = input_data.symptoms.strip()
    if not symptoms:
        raise HTTPException(status_code=400, detail="Symptoms cannot be empty.")

    # Fallback mock if model failed to load
    if model is None or tokenizer is None:
        return {
            "disease": "Model not loaded (check server logs)",
            "confidence": 0.0,
            "warning": "Model could not be loaded. This is a mock response."
        }

    try:
        inputs = tokenizer(
            symptoms,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

        pred_idx = pred.item()

        # Decode label
        if le is not None:
            try:
                disease = le.inverse_transform([pred_idx])[0]
            except Exception:
                disease = f"Class {pred_idx}"
        else:
            disease = f"Class {pred_idx} (label encoder missing)"

        return {
            "disease": disease,
            "confidence": round(float(conf.item()), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "label_encoder_loaded": le is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
