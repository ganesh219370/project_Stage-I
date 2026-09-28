"""
Preprocessing Pipeline for Heart Disease Prediction
Handles data loading, cleaning, validation, imputation, and feature scaling.
"""

import os
import pandas as pd
import numpy as np
 


FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

TARGET_COLUMN = 'target'

NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

FEATURE_NAMES_READABLE = {
    'age': 'Age (years)',
    'sex': 'Sex (0: Female, 1: Male)',
    'cp': 'Chest Pain Type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic)',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dL)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dL (0: False, 1: True)',
    'restecg': 'Resting ECG (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy)',
    'thalach': 'Maximum Heart Rate Achieved (bpm)',
    'exang': 'Exercise-Induced Angina (0: No, 1: Yes)',
    'oldpeak': 'ST Depression (Oldpeak)',
    'slope': 'Slope of Peak Exercise ST (1: Upsloping, 2: Flat, 3: Downsloping)',
    'ca': 'Major Vessels Colored by Fluoroscopy (0-3)',
    'thal': 'Thalassemia (3: Normal, 6: Fixed Defect, 7: Reversable Defect)'
}


def get_data_path():
    """Returns absolute path to dataset CSV."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data', 'heart.csv')


def load_raw_dataset(filepath=None):
    """Loads dataset from CSV file and performs basic integrity checks."""
    if filepath is None:
        filepath = get_data_path()
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Heart disease dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    return df


def create_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer that:
    1. Imputes missing numeric values with median, scales with StandardScaler.
    2. Imputes missing categorical values with most frequent, passes through.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    return preprocessor


def validate_input_data(data_dict):
    """
    Validates patient clinical parameter input data against acceptable ranges.
    Returns (is_valid, cleaned_dict_or_errors).
    """
    errors = {}
    cleaned = {}

    # Age validation
    try:
        age = float(data_dict.get('age', 0))
        if not (18 <= age <= 100):
            errors['age'] = 'Age must be between 18 and 100 years.'
        else:
            cleaned['age'] = age
    except (ValueError, TypeError):
        errors['age'] = 'Invalid age value.'

    # Sex validation (0: Female, 1: Male)
    try:
        sex = int(data_dict.get('sex', -1))
        if sex not in (0, 1):
            errors['sex'] = 'Please select a valid gender option.'
        else:
            cleaned['sex'] = float(sex)
    except (ValueError, TypeError):
        errors['sex'] = 'Invalid sex value.'

    # Chest Pain Type (1, 2, 3, 4)
    try:
        cp = int(data_dict.get('cp', -1))
        if cp not in (1, 2, 3, 4):
            errors['cp'] = 'Please select a valid chest pain classification.'
        else:
            cleaned['cp'] = float(cp)
    except (ValueError, TypeError):
        errors['cp'] = 'Invalid chest pain value.'

    # Resting BP (60 to 240 mm Hg)
    try:
        trestbps = float(data_dict.get('trestbps', 0))
        if not (60 <= trestbps <= 240):
            errors['trestbps'] = 'Resting blood pressure must be between 60 and 240 mm Hg.'
        else:
            cleaned['trestbps'] = trestbps
    except (ValueError, TypeError):
        errors['trestbps'] = 'Invalid resting blood pressure.'

    # Cholesterol (100 to 600 mg/dL)
    try:
        chol = float(data_dict.get('chol', 0))
        if not (100 <= chol <= 600):
            errors['chol'] = 'Serum cholesterol must be between 100 and 600 mg/dL.'
        else:
            cleaned['chol'] = chol
    except (ValueError, TypeError):
        errors['chol'] = 'Invalid cholesterol value.'

    # Fasting blood sugar (0: <=120 mg/dl, 1: >120 mg/dl)
    try:
        fbs = int(data_dict.get('fbs', -1))
        if fbs not in (0, 1):
            errors['fbs'] = 'Please select valid fasting blood sugar status.'
        else:
            cleaned['fbs'] = float(fbs)
    except (ValueError, TypeError):
        errors['fbs'] = 'Invalid fasting blood sugar value.'

    # Resting ECG (0, 1, 2)
    try:
        restecg = int(data_dict.get('restecg', -1))
        if restecg not in (0, 1, 2):
            errors['restecg'] = 'Please select a valid resting ECG result.'
        else:
            cleaned['restecg'] = float(restecg)
    except (ValueError, TypeError):
        errors['restecg'] = 'Invalid resting ECG value.'

    # Max Heart Rate (thalach: 60 to 230 bpm)
    try:
        thalach = float(data_dict.get('thalach', 0))
        if not (60 <= thalach <= 230):
            errors['thalach'] = 'Maximum heart rate must be between 60 and 230 bpm.'
        else:
            cleaned['thalach'] = thalach
    except (ValueError, TypeError):
        errors['thalach'] = 'Invalid maximum heart rate.'

    # Exercise Induced Angina (0: No, 1: Yes)
    try:
        exang = int(data_dict.get('exang', -1))
        if exang not in (0, 1):
            errors['exang'] = 'Please select valid exercise angina status.'
        else:
            cleaned['exang'] = float(exang)
    except (ValueError, TypeError):
        errors['exang'] = 'Invalid exercise angina value.'

    # ST Depression / Oldpeak (0.0 to 7.0)
    try:
        oldpeak = float(data_dict.get('oldpeak', -1))
        if not (0.0 <= oldpeak <= 7.0):
            errors['oldpeak'] = 'ST depression (oldpeak) must be between 0.0 and 7.0 mm.'
        else:
            cleaned['oldpeak'] = oldpeak
    except (ValueError, TypeError):
        errors['oldpeak'] = 'Invalid ST depression value.'

    # ST Slope (1: Upsloping, 2: Flat, 3: Downsloping)
    try:
        slope = int(data_dict.get('slope', -1))
        if slope not in (1, 2, 3):
            errors['slope'] = 'Please select a valid ST slope value.'
        else:
            cleaned['slope'] = float(slope)
    except (ValueError, TypeError):
        errors['slope'] = 'Invalid slope value.'

    # Major Vessels (0, 1, 2, 3)
    try:
        ca = int(data_dict.get('ca', -1))
        if ca not in (0, 1, 2, 3):
            errors['ca'] = 'Number of major vessels must be between 0 and 3.'
        else:
            cleaned['ca'] = float(ca)
    except (ValueError, TypeError):
        errors['ca'] = 'Invalid major vessels value.'

    # Thalassemia (3: Normal, 6: Fixed defect, 7: Reversable defect)
    try:
        thal = int(data_dict.get('thal', -1))
        if thal not in (3, 6, 7):
            errors['thal'] = 'Please select a valid Thalassemia status.'
        else:
            cleaned['thal'] = float(thal)
    except (ValueError, TypeError):
        errors['thal'] = 'Invalid thalassemia value.'

    if errors:
        return False, errors
    return True, cleaned
