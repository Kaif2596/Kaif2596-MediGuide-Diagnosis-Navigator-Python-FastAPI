import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter

# Load the datasets
try:
    df_dataset = pd.read_csv('cleaned dataset.csv')
    df_dataset.columns = df_dataset.columns.str.strip()
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure 'Disease and symptoms dataset.csv' is in the correct directory.")
    exit()

# --- Helper: normalize a symptom string ---
def clean(s):
    """Strip whitespace and replace underscores with spaces."""
    if pd.isna(s):
        return None
    return str(s).strip().replace('_', ' ')

# --- Data Cleaning & Mapping Preparation ---

# Detect disease column dynamically ('diseases' vs 'Disease')
disease_col = 'diseases' if 'diseases' in df_dataset.columns else 'Disease'

# Detect format: old format has 'Symptom_' prefix columns; new format has individual symptom columns
has_symptom_prefix = any(c.startswith('Symptom_') for c in df_dataset.columns)

all_symptoms = set()

if has_symptom_prefix:
    symptom_cols = [c for c in df_dataset.columns if c.startswith('Symptom_')]
    for col in symptom_cols:
        for s in df_dataset[col].dropna().unique():
            cleaned = clean(s)
            if cleaned:
                all_symptoms.add(cleaned.lower())
else:
    symptom_cols = [c for c in df_dataset.columns if c != disease_col]
    for col in symptom_cols:
        cleaned = clean(col)
        if cleaned:
            all_symptoms.add(cleaned.lower())

# --- Feature Engineering ---

# Sorted list → consistent index mapping
sorted_symptoms = sorted(list(all_symptoms))
symptom_to_int = {symptom: i for i, symptom in enumerate(sorted_symptoms)}
num_symptoms = len(sorted_symptoms)

print(f"Total unique symptoms: {num_symptoms}")

# Build feature matrix X and label vector y
X = []
y = []

for _, row in df_dataset.iterrows():
    feature_vector = [0] * num_symptoms
    for col in symptom_cols:
        if has_symptom_prefix:
            raw = row[col]
            symptom = clean(raw)
            if symptom:
                symptom = symptom.lower()
                if symptom in symptom_to_int:
                    idx = symptom_to_int[symptom]
                    feature_vector[idx] = 1
        else:
            # New format: column name is the symptom, cell value is 1 if present
            if row[col] == 1:
                symptom = clean(col)
                if symptom:
                    symptom = symptom.lower()
                    if symptom in symptom_to_int:
                        idx = symptom_to_int[symptom]
                        feature_vector[idx] = 1
    X.append(feature_vector)
    y.append(str(row[disease_col]).strip())

# --- Model Training ---

# FIX: Convert X and y to memory-efficient NumPy arrays before splitting
import numpy as np
X = np.array(X, dtype=np.int16)  # Uses drastically less RAM than a Python list
y = np.array(y)

# Check if any disease has fewer than 2 examples. If so, disable stratification.
class_counts = Counter(y)
if min(class_counts.values()) < 2:
    print("Warning: Some diseases have only 1 sample. Disabling stratification in train_test_split to avoid errors.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# Configured to prevent tree explosion and keep memory low
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=1              # Changed to 1 to prevent multi-threading memory overhead
)

model.fit(X_train, y_train)

# --- Evaluation ---

train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy  = accuracy_score(y_test,  model.predict(X_test))

print(f"Accuracy on Training data: {train_accuracy:.4f}")
print(f"Accuracy on Test data:     {test_accuracy:.4f}")

# --- Save Artifacts ---

joblib.dump(model, 'trained_model.pkl')
print("trained_model.pkl saved.")

joblib.dump(symptom_to_int, 'symptom_to_int.pkl')
print("symptom_to_int.pkl saved.")

# Save disease descriptions safely
descriptions = {}
try:
    with open('symptom_Description.csv', 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                disease = parts[0].strip().replace('_', ' ')
                description = parts[1].strip()
                descriptions[disease] = description
    joblib.dump(descriptions, 'disease_descriptions.pkl')
    print("disease_descriptions.pkl saved.")
except FileNotFoundError:
    print("Warning: symptom_Description.csv not found.")

# Save disease precautions safely
try:
    precautions_df = pd.read_csv('symptom_precaution.csv')
    precautions_df.columns = precautions_df.columns.str.strip()
    prec_disease_col = 'diseases' if 'diseases' in precautions_df.columns else ('Disease' if 'Disease' in precautions_df.columns else precautions_df.columns[0])
    disease_precautions = {}
    for _, row in precautions_df.iterrows():
        disease = str(row[prec_disease_col]).strip().replace('_', ' ')
        precautions = [
            str(row[f'Precaution_{i}']).strip()
            for i in range(1, 5)
            if f'Precaution_{i}' in precautions_df.columns and pd.notna(row[f'Precaution_{i}'])
        ]
        disease_precautions[disease] = precautions
    joblib.dump(disease_precautions, 'disease_precautions.pkl')
    print("disease_precautions.pkl saved.")
except FileNotFoundError:
    print("Warning: symptom_precaution.csv not found.")