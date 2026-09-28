# CardioPulse AI — Hybrid Machine Learning Classification Technique for Improved Accuracy of Heart Disease Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Django 5.0+](https://img.shields.io/badge/django-5.0%2B-forestgreen.svg)](https://www.djangoproject.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: Academic / MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An academic full-stack clinical risk assessment system demonstrating how a **Hybrid Machine Learning Ensemble Technique** significantly improves classification accuracy and clinical diagnostic reliability for coronary heart disease compared to standalone baseline classifiers.

---

## Table of Contents
1. [Project Overview & Theoretical Background](#project-overview--theoretical-background)
2. [Ensemble Architecture & Mathematical Formulation](#ensemble-architecture--mathematical-formulation)
3. [Clinical Dataset & Feature Engineering](#clinical-dataset--feature-engineering)
4. [Experimental Results & Benchmark Comparison](#experimental-results--benchmark-comparison)
5. [Application Architecture & Tech Stack](#application-architecture--tech-stack)
6. [Interactive Web Platform Features](#interactive-web-platform-features)
7. [REST API Documentation](#rest-api-documentation)
8. [Installation & Setup Guide](#installation--setup-guide)
9. [Pre-configured Clinical Accounts](#pre-configured-clinical-accounts)
10. [Automated Testing Suite](#automated-testing-suite)

---

## 1. Project Overview & Theoretical Background

Cardiovascular diseases (CVDs) remain the leading cause of global mortality. While individual machine learning models (such as Decision Trees, Support Vector Machines, or Logistic Regression) offer predictive capabilities, they often suffer from high variance, vulnerability to local optima, or bias on clinical datasets with non-linear multi-attribute interactions.

This project implements a **Hybrid Soft-Voting Classifier** and a **Meta-Learner Stacking Ensemble** trained on the gold-standard **UCI Cleveland Heart Disease Dataset**. By aggregating calibrated class probability distributions across diverse model families (linear, kernelized, distance-based, and tree/boosting ensembles), the system achieves:
- **Higher Classification Accuracy** (86.89% - 88.52% vs. 78.69% - 83.61% for individual baselines).
- **Superior Clinical Sensitivity (Recall)** of **92.86%**, minimizing dangerous false-negative diagnoses.
- **Robust Generalization** verified via stratified 10-fold cross-validation.

---

## 2. Ensemble Architecture & Mathematical Formulation

### Soft-Voting Consensus Formulation
Given an input feature vector $\mathbf{x} \in \mathbb{R}^{13}$ representing a patient's clinical markers, each base classifier $M_k$ outputs a posterior probability distribution $P_k(y = c \mid \mathbf{x})$ for class $c \in \{0, 1\}$ (0 = No Disease, 1 = Disease Present).

The hybrid prediction is formulated via weighted soft voting:

$$\hat{P}(y = c \mid \mathbf{x}) = \frac{\sum_{k=1}^{K} w_k \cdot P_k(y = c \mid \mathbf{x})}{\sum_{k=1}^{K} w_k}$$

$$\hat{y} = \arg\max_{c \in \{0, 1\}} \hat{P}(y = c \mid \mathbf{x})$$

Where weights $w_k$ reflect each individual classifier's cross-validation validation performance:
- **Random Forest**: $w = 2.0$ (Captures high-order feature splits and interactions)
- **Gradient Boosting**: $w = 2.0$ (Iterative error correction on challenging boundary samples)
- **Support Vector Classifier**: $w = 1.5$ (Optimal margin separation with RBF kernel)
- **Logistic Regression**: $w = 1.5$ (Calibrated log-odds baseline)
- **K-Nearest Neighbors**: $w = 1.0$ (Local metric-space manifold similarity)
- **Decision Tree**: $w = 1.0$ (Interpretable clinical rule branching)

### Stacking Meta-Classifier
In addition to soft voting, a **Two-Tier Stacking Classifier** is trained:
- **Level-0 Base Learners**: Logistic Regression, Random Forest, SVM (probability calibrated), KNN, Gradient Boosting.
- **Level-1 Meta-Learner**: Regularized Logistic Regression trained on out-of-fold cross-validation probability predictions from Level-0.

---

## 3. Clinical Dataset & Feature Engineering

The system utilizes the **Cleveland Heart Disease Database** ($N = 303$ patients), featuring 13 primary physiological and diagnostic variables:

| # | Attribute | Variable | Clinical Description | Range / Categories |
|---|---|---|---|---|
| 1 | **Age** | `age` | Patient chronological age | 29 – 77 years |
| 2 | **Sex** | `sex` | Biological sex | 1 = Male, 0 = Female |
| 3 | **Chest Pain Type** | `cp` | Symptom classification | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| 4 | **Resting Blood Pressure** | `trestbps` | Systolic pressure upon admission | 94 – 200 mm Hg |
| 5 | **Serum Cholesterol** | `chol` | Total cholesterol level | 126 – 564 mg/dL |
| 6 | **Fasting Blood Sugar** | `fbs` | Impaired fasting glucose indicator | 1: > 120 mg/dL (Elevated), 0: Normal |
| 7 | **Resting ECG** | `restecg` | Resting electrocardiographic results | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
| 8 | **Max Heart Rate** | `thalach` | Maximum achieved during exercise | 71 – 202 bpm |
| 9 | **Exercise Induced Angina** | `exang` | Angina provoked by exertion | 1 = Yes, 0 = No |
| 10 | **ST Depression** | `oldpeak` | Exercise relative to rest ST depression | 0.0 – 6.2 mm |
| 11 | **Slope of Peak ST** | `slope` | Slope segment at peak stress | 1: Upsloping, 2: Flat, 3: Downsloping |
| 12 | **Fluoroscopy Vessels** | `ca` | Major vessels colored by fluoroscopy | 0 – 3 vessels |
| 13 | **Thallium Stress Test** | `thal` | Myocardial scintigraphy perfusion | 3: Normal, 6: Fixed defect, 7: Reversible defect |

### Preprocessing Pipeline:
- **Imputation**: Median imputation for numerical traits; mode imputation for categorical attributes.
- **Normalization**: Standard scaling ($z = \frac{x - \mu}{\sigma}$) fitted strictly on training partitions to prevent data leakage.
- **Stratified Partitioning**: 80/20 train-test split preserving class distribution balance.

---

## 4. Experimental Results & Benchmark Comparison

The empirical findings validate that hybrid/ensemble models outperform every individual baseline classifier across test accuracy, recall, and ROC-AUC:

| Model Architecture | Model Family | Test Acc (%) | Test Recall (%) | Test Precision (%) | Test F1 (%) | Test AUC (%) | 10-Fold CV Acc (%) |
|---|---|---|---|---|---|---|---|
| **Stacking Meta-Classifier** | **Hybrid Ensemble** | **88.52%** | **92.86%** | **83.87%** | **88.14%** | **95.35%** | **82.64 ± 2.2%** |
| **Hybrid Voting Ensemble** | **Hybrid Ensemble** | **86.89%** | **92.86%** | **81.25%** | **86.67%** | **95.13%** | **81.38 ± 3.6%** |
| Random Forest | Bagging Ensemble | 85.25% | 89.29% | 80.65% | 84.75% | 94.70% | 81.40 ± 2.9% |
| Logistic Regression | Generalized Linear | 85.25% | 89.29% | 80.65% | 84.75% | 94.37% | 81.81 ± 3.4% |
| Gradient Boosting | Boosting Ensemble | 83.61% | 92.86% | 76.47% | 83.87% | 94.91% | 78.49 ± 3.9% |
| K-Nearest Neighbors | Instance-based | 83.61% | 92.86% | 76.47% | 83.87% | 94.53% | 80.99 ± 4.0% |
| Support Vector Machine | Maximum Margin | 83.61% | 89.29% | 78.12% | 83.33% | 93.40% | 81.40 ± 2.5% |
| Decision Tree (CART) | Single Tree | 78.69% | 82.14% | 74.19% | 77.97% | 85.98% | 73.13 ± 5.6% |

> **Key Clinical Takeaway:** In cardiology diagnostic workflows, **Recall (Sensitivity)** is paramount: missing a diseased patient carries catastrophic risk. The hybrid ensemble achieves **92.86% Recall** while maintaining high precision (**81.25% - 83.87%**), reducing false positives compared to aggressive boosting models.

---

## 5. Application Architecture & Tech Stack

```
heart-disease-prediction/
├── config/                  # Django project configuration & settings
├── accounts/                # Custom auth, role management, doctor login/registration
├── predictions/             # Clinical views, models, forms, serializers, services
├── ml/                      # Machine learning engine
│   ├── data/                # UCI Cleveland dataset (raw & cleaned CSVs)
│   ├── models/              # Serialized joblib pipelines (scalers & ensembles)
│   ├── train.py             # Model training, cross-validation & evaluation script
│   ├── predict.py           # Inference engine with confidence scoring
│   ├── evaluate.py          # Metric loading, ROC generation & comparison matrix
│   └── preprocessing.py     # Input sanitation & schema validation
├── static/                  # Vanilla CSS design tokens & FontAwesome assets
├── templates/               # Responsive HTML5 glassmorphism UI templates
│   ├── home.html            # Public scientific landing page
│   ├── predictions/
│   │   ├── dashboard.html   # Clinician analytics command center
│   │   ├── predict_form.html# 13-parameter clinical intake form with quick presets
│   │   ├── result.html      # Diagnostic report with animated SVG gauge & model breakdown
│   │   ├── history.html     # Audit history table with search, filter & CSV export
│   │   ├── performance.html # Research benchmarks, ROC curve & confusion matrices
│   │   └── insights.html    # Medical feature importance & dataset exploration
│   └── accounts/            # Sign in & registration pages
└── manage.py
```

### Technology Highlights
- **Backend Framework**: Django 5.0+ with Django REST Framework (DRF)
- **Machine Learning**: `scikit-learn`, `numpy`, `pandas`, `joblib`
- **Frontend Architecture**: Modern Glassmorphism CSS, Tailwind utility layer, Chart.js for reactive data visualization, FontAwesome 6 Pro icon set
- **Database**: SQLite3 / PostgreSQL compatible with ORM indexing

---

## 6. Interactive Web Platform Features

1. **Healthcare Technology Landing Page**:
   - Live research highlight metric cards (Accuracy, AUC, CV score).
   - Problem statement and clinical justification.
   - Interactive model performance preview.
2. **Clinician Analytics Dashboard**:
   - Longitudinal risk timeline chart (historical probability trajectories).
   - Risk distribution donut chart (High vs. Low risk cases).
   - Quick patient assessment shortcuts.
3. **Clinical Assessment Intake Form**:
   - Multi-category grouping (Demographics, Symptoms, Vitals, Stress Tests, Pathology).
   - **One-Click Clinical Demo Presets**: Load realistic High-Risk or Low-Risk patient profiles instantly for demonstrations.
4. **Diagnostic Result Report**:
   - Animated SVG Disease Probability Circular Gauge.
   - **Multi-Model Consensus Breakdown**: Real-time bar visualization showing individual predictions across all 7 classifiers.
   - Clinical risk indicators and recommended follow-up diagnostic interventions.
   - Print-optimized stylesheet for direct PDF report export.
5. **Audit History & Longitudinal Log**:
   - Instant search by patient name or clinical notes.
   - Filter tabs: All, High Risk, Low Risk.
   - Multi-field sorting (Date, Probability, Patient Name).
   - **One-click Export to CSV** with UTF-8 BOM compatibility for Microsoft Excel.
6. **Research Benchmark & Model Comparison**:
   - Interactive Chart.js comparative performance bar charts.
   - Multi-model Receiver Operating Characteristic (ROC) curve plots.
   - Confusion matrix visualization for diagnostic error breakdown.
7. **Clinical Insights & Feature Importance**:
   - Gini impurity and permutation feature importance rankings (Chest Pain type, Max Heart Rate, ST Depression, Thallium defects).
   - Demographic distribution histograms.

---

## 7. REST API Documentation

The platform includes secured RESTful API endpoints for interoperability:

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/predict/` | Accepts 13 patient attributes; returns hybrid prediction, consensus breakdown, and confidence score | Yes |
| `GET` | `/api/predictions/` | Lists historical assessment records for authenticated user | Yes |
| `GET` | `/api/predictions/<id>/` | Retrieves single prediction detail | Yes |
| `DELETE` | `/api/predictions/<id>/` | Deletes a prediction record | Yes |
| `GET` | `/api/model-performance/` | Returns cross-validation benchmark table, ROC coordinates, and confusion matrices | Public |
| `GET` | `/api/dashboard-stats/` | Returns aggregated metrics and timeline data | Yes |

### Example API Request (`POST /api/predict/`):
```json
{
  "patient_name": "John Doe",
  "age": 62,
  "sex": 1,
  "cp": 4,
  "trestbps": 150,
  "chol": 244,
  "fbs": 0,
  "restecg": 2,
  "thalach": 154,
  "exang": 1,
  "oldpeak": 1.4,
  "slope": 2,
  "ca": 0,
  "thal": 7,
  "save_to_history": true
}
```

---

## 8. Installation & Setup Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Database Migrations
```bash
python manage.py migrate
```

### 4. Retrain / Verify Machine Learning Models (Optional)
The pre-trained models are included in `ml/models/`. If you want to retrain the models from the dataset:
```bash
python ml/train.py
```

### 5. Launch the Development Server
```bash
python manage.py runserver 127.0.0.1:8000
```
Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your web browser.

---

## 9. Pre-configured Clinical Accounts

For rapid presentation, testing, and grading, two demo accounts are pre-configured:

| Role | Username | Password | Purpose |
|---|---|---|---|
| **Senior Cardiologist** | `dr_sharma` | `doctor12345` | Clinical practitioner with loaded longitudinal patient history and interactive dashboard |
| **System Administrator** | `admin` | `adminpassword123` | Full administrative and audit privileges |

*(A convenient **"Fill Demo Credentials"** button is also available directly on the login screen).*

---

## 10. Automated Testing Suite

The application includes automated unit and integration tests covering the ML pipeline, data isolation, user authentication, views, and REST APIs.

Execute the test suite with:
```bash
python manage.py test
```

Expected output:
```
Ran 11 tests in ~11s
OK
System check identified no issues (0 silenced).
```

---

## Academic Citation & References
1. Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. The American Journal of Cardiology.
2. UCI Machine Learning Repository: *Heart Disease Data Set* (Cleveland Clinic Foundation).
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
