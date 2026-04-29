"""
Run this from inside the backend/ directory:
    cd backend
    python init_labels.py
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.joblib')

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
le.fit(df['Disease'])

joblib.dump(le, LABEL_ENCODER_PATH)
print(f"Label encoder saved to {LABEL_ENCODER_PATH}")
print(f"Diseases ({len(le.classes_)}): {list(le.classes_)}")
