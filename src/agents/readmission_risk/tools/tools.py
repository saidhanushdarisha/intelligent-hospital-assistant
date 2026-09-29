import joblib
import pandas as pd
from langchain_core.tools import tool

# Load model and encoders
readmission_risk_model = joblib.load(
    "src/agents/readmission_risk/tools/readmission_risk.pkl"
)

model = readmission_risk_model["model"]
gender_encoder = readmission_risk_model["gender_encoder"]
condition_encoder = readmission_risk_model["condition_encoder"]
procedure_encoder = readmission_risk_model["procedure_encoder"]
outcome_encoder = readmission_risk_model["outcome_encoder"]
readmission_encoder = readmission_risk_model["readmission_encoder"]


@tool
def predict_readmission(
    Age: int,
    Gender: str,
    Condition: str,
    Procedure: str,
    Cost: float,
    Length_of_Stay: int,
    Outcome: str,
    Satisfaction: int,
) -> dict:
    """Predict readmission risk of a patient using health and demographic details."""

    # Transform categorical features individually to avoid mutating original args
    encoded_gender = gender_encoder.transform([Gender])[0]
    encoded_condition = condition_encoder.transform([Condition])[0]
    encoded_procedure = procedure_encoder.transform([Procedure])[0]
    encoded_outcome = outcome_encoder.transform([Outcome])[0]

    # Construct clean dataframe matching model features
    patient_data = {
        "Age": int(Age),
        "Gender": encoded_gender,
        "Condition": encoded_condition,
        "Procedure": encoded_procedure,
        "Cost": float(Cost),
        "Length_of_Stay": int(Length_of_Stay),
        "Outcome": encoded_outcome,
        "Satisfaction": int(Satisfaction),
    }

    input_df = pd.DataFrame([patient_data])

    # Run inference
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = round(float(max(probability)) * 100, 2)

    # Decode prediction result
    decoded_result = readmission_encoder.inverse_transform([prediction])[0]

    # Convert NumPy scalar to native Python string/int to avoid JSON serialization errors
    if hasattr(decoded_result, "item"):
        decoded_result = decoded_result.item()
    else:
        decoded_result = str(decoded_result)

    return {"Readmission": decoded_result, "Confidence": float(confidence)}


READMISSION_RISK_TOOLS = [predict_readmission]