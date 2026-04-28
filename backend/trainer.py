import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Preprocess: Combine symptoms into a single string
symptom_cols = [c for c in df.columns if 'Symptom' in c]
df['combined_symptoms'] = df[symptom_cols].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['Disease'])
num_labels = len(le.classes_)

# Save label encoder for inference
import joblib
joblib.dump(le, 'label_encoder.joblib')

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer
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
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    train_dataset = SymptomDataset(
        texts=train_df['combined_symptoms'].to_list(),
        labels=train_df['label'].to_list(),
        tokenizer=tokenizer
    )
    
    val_dataset = SymptomDataset(
        texts=val_df['combined_symptoms'].to_list(),
        labels=val_df['label'].to_list(),
        tokenizer=tokenizer
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
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
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')

if __name__ == "__main__":
    # Note: In a real scenario, you'd run this. For the zip, we provide the code.
    print("Training script ready. Run this to fine-tune BioBERT on your dataset.")
