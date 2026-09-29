import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('src/agents/model_drifting/datasets/hospital_data_analysis.csv')

df.isnull().sum()

df.duplicated().sum()

sns.countplot(x='Readmission',data=df)
plt.title('Readmission Distribution')
plt.show()

df.drop("Patient_ID",axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# le=LabelEncoder()

# catergorical_columns=['Gender','Condition','Procedure','Outcome','Readmission']

# for col in catergorical_columns:
#     df[col]=le.fit_transform(df[col])

# df.head()

gender_encoder=LabelEncoder()
condition_encoder=LabelEncoder()
procedure_encoder=LabelEncoder()
outcome_encoder=LabelEncoder()
readmission_encoder=LabelEncoder()

df['Gender']=gender_encoder.fit_transform(df['Gender'])
df['Condition']=condition_encoder.fit_transform(df['Condition'])
df['Procedure']=procedure_encoder.fit_transform(df['Procedure'])
df['Outcome']=outcome_encoder.fit_transform(df['Outcome'])
df['Readmission']=readmission_encoder.fit_transform(df['Readmission'])
df.head()

X=df.drop("Readmission",axis=1)
y=df['Readmission']

X_train,X_temp,y_train,y_temp=train_test_split(
    X,y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
X_val,X_test,y_val,y_test=train_test_split(
    X_temp,y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("Training Samples: ",len(X_train))
print("Validation Samples: ",len(X_val))
print("Testing Samples: ",len(X_test))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier

xgb_model=XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train,y_train)

y_val_pred=xgb_model.predict(X_val)
y_test_pred=xgb_model.predict(X_test)

# Validation

print("Validation Accuracy: ",accuracy_score(y_val,y_val_pred))
print("Validation Precision: ",precision_score(y_val,y_val_pred))
print("Validation Recall: ",recall_score(y_val,y_val_pred))
print("Validation F1 Score: ",f1_score(y_val,y_val_pred))

# Testing

print("Testing Accuracy: ",accuracy_score(y_test,y_test_pred))
print("Testing Precision: ",precision_score(y_test,y_test_pred))
print("Testing Recall: ",recall_score(y_test,y_test_pred))
print("Testing F1 Score: ",f1_score(y_test,y_test_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No','Yes'],
    yticklabels=['No','Yes']
)
plt.xlabel('predicated')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Data')
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val,y_val_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['No','Yes'],
    yticklabels=['No','Yes']
)
plt.xlabel('predicated')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Validated Data')
plt.show()

from xgboost import plot_importance

plot_importance(xgb_model,max_num_features=10)
plt.show()

print("\n Validation Classification Report\n")
print(classification_report(y_val,y_val_pred))

print("\n Testing Classification Report\n")
print(classification_report(y_test,y_test_pred))

import joblib

artifacts = {
    "model": xgb_model,
    "gender_encoder": gender_encoder,
    "condition_encoder": condition_encoder,
    "procedure_encoder": procedure_encoder,
    "outcome_encoder": outcome_encoder,
    "readmission_encoder": readmission_encoder
}

joblib.dump(artifacts, "readmission_model.pkl")

print("Readmission Model Saved Successfully")

stats={}

numerical_features=[
    "Age",
    "Cost",
    "Length_of_Stay",
    "Satisfaction"
]

for feature in numerical_features:
    stats[feature]={
        "mean": float(X_train[feature].mean()),
        "std": float(X_train[feature].std()),
        "min": float(X_train[feature].min()),
        "max": float(X_train[feature].max())        
        }
    
joblib.dump(stats, "readmission_stats.pkl")

stats=joblib.load("readmission_stats.pkl")
print(stats)

def check_readmission_drift(patient_data, stats):

    drift_detected = False

    drift_report = []

    for feature in stats:

        value = patient_data[feature]

        mean = stats[feature]["mean"]
        std = stats[feature]["std"]

        lower_limit = mean - (3 * std)
        upper_limit = mean + (3 * std)

        if value < lower_limit or value > upper_limit:

            drift_detected = True

            drift_report.append({
                "Feature": feature,
                "Input": value,
                "Expected Range": f"{round(lower_limit,2)} - {round(upper_limit,2)}"
            })

    return drift_detected, drift_report

patient = {
    "Age": 45,
    "Cost": 15000,
    "Length_of_Stay": 5,
    "Satisfaction": 4
}

drift, report = check_readmission_drift(patient, stats)

print("Drift Detected :", drift)

print(report)

patient = {
    "Age": 150,
    "Cost": 500000,
    "Length_of_Stay": 80,
    "Satisfaction": 10
}

drift, report = check_readmission_drift(patient, stats)

print("Drift Detected :", drift)

for item in report:

    print(item)


plt.figure(figsize=(7,5))

sns.histplot(X_train["Age"], bins=15, kde=True, color="skyblue")

plt.axvline(
    patient["Age"],
    color="red",
    linestyle="--",
    linewidth=3,
    label="Patient Age"
)

plt.title("Model Drift Monitoring - Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()

plt.show()

plt.figure(figsize=(6,5))

sns.boxplot(y=X_train["Age"], color="lightgreen")

plt.scatter(
    0,
    patient["Age"],
    color="red",
    s=150,
    label="Patient Age"
)

plt.title("Age Drift Detection")

plt.legend()

plt.show()