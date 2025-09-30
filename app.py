from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import dice_ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import glob

app = Flask(__name__)

outcomes_df = pd.read_csv('archive/Outcomes-a.txt')
all_files = glob.glob(os.path.join('archive/set-a/set-a', "*.txt"))

all_records_list = []
for filename in all_files:
    patient_df = pd.read_csv(filename)
    patient_series = patient_df.groupby('Parameter')['Value'].mean()
    record_data = patient_series.to_dict()
    all_records_list.append(record_data)

features_df = pd.DataFrame(all_records_list)
full_df = pd.merge(features_df, outcomes_df, on='RecordID')


cols_to_replace = full_df.columns.difference(['In-hospital_death'])
full_df[cols_to_replace] = full_df[cols_to_replace].replace(-1, np.nan)

for col in full_df.columns:
    if full_df[col].isnull().any():
        full_df[col].fillna(full_df[col].mean(), inplace=True)

features = ['ALP', 'ALT', 'AST', 'Age', 'BUN', 'Bilirubin', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Gender', 'Glucose', 'HCO3', 'HCT', 'HR', 'Height', 'ICUType', 'K', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'SysABP', 'Temp', 'Urine', 'WBC', 'Weight', 'pH', 'Albumin', 'RespRate', 'Lactate', 'SaO2', 'TroponinT', 'TroponinI', 'Cholesterol', 'SAPS-I', 'SOFA']
target = 'In-hospital_death'

X = full_df[features]
y = full_df[target]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, 'model.pkl')

model = joblib.load('model.pkl')

# If your outcomes file has 'Survival' instead of 'In-hospital_death'
if 'Survival' in full_df.columns:
    full_df.rename(columns={'Survival': 'In-hospital_death'}, inplace=True)

# Drop 'RecordID' and 'Length_of_stay' from full_df for DiCE if present
drop_cols = [col for col in ['RecordID', 'Length_of_stay'] if col in full_df.columns]
dice_df = full_df.drop(columns=drop_cols)

# Ensure dice_df only has features + target columns, and features are in the same order
dice_df = dice_df[features + [target]]

@app.route('/')
def home():
    random_record = dice_df.sample(1, random_state=42)
    input_df = random_record[features]  # Ensure correct columns and order

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    d = dice_ml.Data(dataframe=dice_df, continuous_features=features, outcome_name=target)
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    dice_exp = exp.generate_counterfactuals(input_df, total_CFs=3, desired_class="opposite")

    return render_template(
        'index.html',
        features=features,
        prediction=prediction,
        probability=probability,
        counterfactuals=dice_exp.cf_examples_list[0].final_cfs_df
    )

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[f]) for f in features]
    input_df = pd.DataFrame([input_data], columns=features)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    d = dice_ml.Data(dataframe=dice_df, continuous_features=features, outcome_name=target)
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    
    dice_exp = exp.generate_counterfactuals(input_df, total_CFs=3, desired_class="opposite")

    return render_template('index.html', features=features, prediction=prediction, probability=probability, counterfactuals=dice_exp.cf_examples_list[0].final_cfs_df)

if __name__ == '__main__':
    app.run(debug=True)