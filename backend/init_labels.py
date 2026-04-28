import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset to get all disease labels
df = pd.read_csv('data/dataset.csv')
le = LabelEncoder()
le.fit(df['Disease'])

# Save label encoder
joblib.dump(le, 'label_encoder.joblib')
print(f"Label encoder initialized with {len(le.classes_)} diseases.")
