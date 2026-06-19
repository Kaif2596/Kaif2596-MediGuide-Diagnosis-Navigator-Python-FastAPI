# Import necessary libraries
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import Optional

# Create a FastAPI app
app = FastAPI(title="MediGuide-Diagnosis Navigator")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the ML model and symptom mappings
try:
    model          = joblib.load('trained_model.pkl')
    symptom_to_int = joblib.load('symptom_to_int.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or dictionaries: {e}")
    print("Please run Train.py to generate the necessary files.")
    exit()

# Load disease descriptions directly from CSV (picks up edits on server restart)
def load_descriptions_from_csv(file_path: str) -> dict:
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df.columns = df.columns.str.strip()
        disease_col = 'diseases' if 'diseases' in df.columns else ('Disease' if 'Disease' in df.columns else df.columns[0])
        desc_col = 'Description' if 'Description' in df.columns else df.columns[1]
        df[disease_col] = df[disease_col].str.strip().str.replace('_', ' ').str.lower()
        return dict(zip(df[disease_col], df[desc_col].str.strip()))
    except (FileNotFoundError, KeyError):
        print(f"Warning: {file_path} not found or invalid format.")
        return {}

# Load disease precautions directly from CSV (picks up edits on server restart)
def load_precautions_from_csv(file_path: str) -> dict:
    precautions = {}
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        df.columns = df.columns.str.strip()
        disease_col = 'diseases' if 'diseases' in df.columns else ('Disease' if 'Disease' in df.columns else df.columns[0])
        for _, row in df.iterrows():
            disease = str(row[disease_col]).strip().replace('_', ' ').lower()
            precs = [
                str(row[f'Precaution {i}']).strip()
                for i in range(1, 5)
                if f'Precaution {i}' in df.columns and pd.notna(row[f'Precaution {i}'])
            ]
            precautions[disease] = precs
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
    return precautions


disease_descriptions = load_descriptions_from_csv('symptom_Description.csv')
disease_precautions  = load_precautions_from_csv('symptom_precaution.csv')

symptom_list = sorted(list(symptom_to_int.keys()))
num_symptoms = len(symptom_list)

# Define the home route
@app.get('/', response_class=HTMLResponse)
async def index(request: Request, search: Optional[str] = Query(None)):
    filtered_symptoms = symptom_list
    if search:
        filtered_symptoms = [s for s in symptom_list if search.lower() in s.lower()]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"symptoms": filtered_symptoms}
    )

# Define the submit route for form submission
@app.post('/submit', response_class=HTMLResponse)
async def submit(request: Request):
    try:
        form_data = await request.form()
        selected_symptoms = form_data.getlist("symptoms")

        # Guard: require at least one symptom to be selected
        if not selected_symptoms:
            error_html = """
            <html><head><title>Error - MediGuide</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Roboto', sans-serif; background-color: #f5f5dc;
                       display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
                .box { background: #fff; padding: 40px; border-radius: 8px; text-align: center;
                       box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-width: 400px; }
                h2 { color: #dc3545; }
                a { display: inline-block; margin-top: 20px; padding: 10px 24px;
                    background: #007bff; color: #fff; border-radius: 4px; text-decoration: none; }
                a:hover { background: #0056b3; }
            </style></head>
            <body><div class="box">
                <h2>⚠️ No Symptoms Selected</h2>
                <p>Please select at least one symptom before submitting.</p>
                <a href="/">← Go Back</a>
            </div></body></html>
            """
            return HTMLResponse(content=error_html, status_code=400)

        # Create the feature vector
        feature_vector = [0] * num_symptoms
        for symptom in selected_symptoms:
            # Normalize to match training format: strip, no underscores, lowercase
            symptom_clean = symptom.strip().replace('_', ' ').lower()
            if symptom_clean in symptom_to_int:
                symptom_index = symptom_to_int[symptom_clean]
                feature_vector[symptom_index] = 1

        # Convert to numpy array and reshape
        input_data = np.array(feature_vector).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        predicted_disease_lower = str(prediction).lower()
        predicted_disease_display = str(prediction).title()

        # Get description and precautions
        predicted_description = disease_descriptions.get(predicted_disease_lower, "Description not available.")
        predicted_precautions = disease_precautions.get(predicted_disease_lower, [])

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "prediction": predicted_disease_display,
                "description": predicted_description,
                "precautions": predicted_precautions
            }
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HTMLResponse(content=f"<h3>Internal Server Error Debug:</h3><pre>{error_details}</pre>", status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)