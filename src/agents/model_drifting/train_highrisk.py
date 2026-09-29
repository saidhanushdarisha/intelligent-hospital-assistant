import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("src/agents/model_drifting/datasets/Health_Risk_Dataset.csv")

print("Dataset Shape:", df.shape)

df.head()

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

sns.countplot(x="Risk_Level", data=df)

plt.title("Risk Level Distribution")
plt.show()

df.drop("Patient_ID", axis=1, inplace=True)

df.head()

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:", df.shape)

consciousness_encoder = LabelEncoder()

df["Consciousness"] = consciousness_encoder.fit_transform(
    df["Consciousness"]
)

print("Consciousness Classes:")
print(consciousness_encoder.classes_)

risk_encoder = LabelEncoder()

df["Risk_Level"] = risk_encoder.fit_transform(
    df["Risk_Level"]
)

print("Risk Level Classes:")
print(risk_encoder.classes_)

print("\nRisk Level Distribution:\n")
print(df["Risk_Level"].value_counts())

X = df.drop("Risk_Level", axis=1)

y = df["Risk_Level"]

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("Training Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))

print("Precision :", precision_score(
    y_test, y_pred,
    average='weighted'
))

print("Recall :", recall_score(
    y_test, y_pred,
    average='weighted'
))

print("F1 Score :", f1_score(
    y_test, y_pred,
    average='weighted'
))


print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8,5))

sns.barplot(
    x=feature_importance.values,
    y=feature_importance.index
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()

artifacts = {
    "model": model,
    "consciousness_encoder": consciousness_encoder,
    "risk_encoder": risk_encoder
}

joblib.dump(
    artifacts,
    "high_risk_artifacts.pkl"
)

print("Artifacts Model Saved Successfully!")

print("\nConsciousness Classes:")
print(consciousness_encoder.classes_)

print("\nRisk Classes:")
print(risk_encoder.classes_)

stats = {}

numerical_features = [
    "Respiratory_Rate",
    "Oxygen_Saturation",
    "O2_Scale",
    "Systolic_BP",
    "Heart_Rate",
    "Temperature"
]

for feature in numerical_features:

    stats[feature] = {

        "mean": float(X_train[feature].mean()),
        "std": float(X_train[feature].std()),
        "min": float(X_train[feature].min()),
        "max": float(X_train[feature].max())

    }

joblib.dump(
    stats,
    "high_risk_stats.pkl"
)

print("High Risk Statistics Saved Successfully!")


def check_highrisk_drift(patient, stats):

    drift = False
    report = []

    for feature in stats:

        value = patient[feature]

        lower = stats[feature]["mean"] - 3 * stats[feature]["std"]
        upper = stats[feature]["mean"] + 3 * stats[feature]["std"]

        if value < lower or value > upper:

            drift = True

            report.append({
                "Feature": feature,
                "Input": value,
                "Expected Range": f"{lower:.2f} - {upper:.2f}"
            })

    return drift, report


patient = {

    "Respiratory_Rate":22,
    "Oxygen_Saturation":96,
    "O2_Scale":1,
    "Systolic_BP":110,
    "Heart_Rate":92,
    "Temperature":37

}

drift, report = check_highrisk_drift(patient, stats)

print("Drift Detected :", drift)

for item in report:
    print(item)


patient = {

    "Respiratory_Rate":50,
    "Oxygen_Saturation":60,
    "O2_Scale":5,
    "Systolic_BP":250,
    "Heart_Rate":180,
    "Temperature":43

}

drift, report = check_highrisk_drift(patient, stats)

print("Drift Detected :", drift)

for item in report:
    print(item)

plt.figure(figsize=(7,5))

sns.histplot(
    X_train["Heart_Rate"],
    bins=15,
    kde=True,
    color="skyblue"
)

plt.axvline(
    patient["Heart_Rate"],
    color="red",
    linestyle="--",
    linewidth=3,
    label="Patient Heart Rate"
)

plt.title("Model Drift Monitoring - Heart Rate")

plt.xlabel("Heart Rate")

plt.ylabel("Frequency")

plt.legend()

plt.show()



plt.figure(figsize=(6,5))

sns.boxplot(
    y=X_train["Heart_Rate"],
    color="lightgreen"
)

plt.scatter(
    0,
    patient["Heart_Rate"],
    color="red",
    s=150,
    label="Patient Heart Rate"
)

plt.title("Heart Rate Drift Detection")

plt.legend()

plt.show()
