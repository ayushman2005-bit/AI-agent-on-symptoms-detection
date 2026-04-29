import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths — run this script from the backend/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.joblib')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_model')

# Load dataset
df = pd.read_csv(DATA_PATH)

# Combine symptom columns into one string
symptom_cols = [c for c in df.columns if 'Symptom' in c]
df['combined_symptoms'] = df[symptom_cols].apply(
    lambda x: ', '.join(x.dropna().astype(str).str.strip()), axis=1
)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['Disease'])
num_labels = len(le.classes_)
print(f"Number of disease classes: {num_labels}")

# Save label encoder alongside app.py in backend/
joblib.dump(le, LABEL_ENCODER_PATH)
print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

# Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class SymptomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[item]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def train():
    train_dataset = SymptomDataset(
        texts=train_df['combined_symptoms'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer
    )
    val_dataset = SymptomDataset(
        texts=val_df['combined_symptoms'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, 'results'),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(BASE_DIR, 'logs'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    print("Starting BioBERT fine-tuning...")
    train()
