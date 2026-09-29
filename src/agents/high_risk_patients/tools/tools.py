import joblib
import pandas as pd
from langchain_core.tools import tool

# Load model and encoders
high_risk_patients_model = joblib.load("src/agents/high_risk_patients/tools/high_risk_patients.pkl")

model = high_risk_patients_model["model"]
con_encoder = high_risk_patients_model["consciousness_encoder"]
risk_encoder = high_risk_patients_model["risk_encoder"]

@tool
def predict_highrisk(
    Respiratory_Rate: int,
    Oxygen_Saturation: int,
    O2_Scale: int,
    Systolic_BP: int,
    Heart_Rate: int,
    Temperature: float,
    Consciousness: str,
    On_Oxygen: int
) -> dict:
    """Predict risk level of a patient."""
    
    # 1. Safely transform categorical features without mutating args
    encoded_con = con_encoder.transform([Consciousness])[0]

    # 2. Build clean input mapping 
    patient_data = {
        "Respiratory_Rate": int(Respiratory_Rate),
        "Oxygen_Saturation": int(Oxygen_Saturation),
        "O2_Scale": int(O2_Scale),
        "Systolic_BP": int(Systolic_BP),
        "Heart_Rate": int(Heart_Rate),
        "Temperature": float(Temperature),
        "Consciousness": encoded_con,
        "On_Oxygen": int(On_Oxygen)
    }

    input_df = pd.DataFrame([patient_data])

    # 3. Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    confidence = round(float(max(probability)) * 100, 2)
    risk = risk_encoder.inverse_transform([prediction])[0]

    # 4. Cast numpy types back to native python types
    if hasattr(risk, "item"):
        risk = risk.item()
    else:
        risk = str(risk)

    return {
        "Risk_Level": risk,
        "Risk_Score": float(confidence)
    }


HIGH_RISK_PATIENTS_TOOLS = [
    predict_highrisk
]