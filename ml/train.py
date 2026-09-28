"""
Model Training Script for Heart Disease Prediction
Trains 6 individual classification models, a primary Hybrid Soft Voting Classifier,
and a Stacking Classifier. Evaluates all models with 5-fold cross-validation and test metrics,
and serializes artifacts for Django to load.
"""

import os
import sys
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from ml.preprocessing import (
    load_raw_dataset,
    create_preprocessor,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURE_NAMES_READABLE
)


def train_and_evaluate(save_artifacts=True):
    print("=" * 60)
    print("HYBRID MACHINE LEARNING FOR HEART DISEASE PREDICTION")
    print("Starting Training & Model Evaluation Pipeline...")
    print("=" * 60)

    # 1. Load Data
    df = load_raw_dataset()
    total_samples = len(df)
    features_count = len(FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    class_counts = y.value_counts().to_dict()
    neg_count = int(class_counts.get(0, 0))
    pos_count = int(class_counts.get(1, 0))

    print(f"Dataset Loaded: {total_samples} samples, {features_count} clinical features.")
    print(f"Class Distribution: {neg_count} Low Risk (0), {pos_count} Heart Disease Detected (1).")

    # 2. Stratified Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train Set: {len(X_train)} samples | Test Set: {len(X_test)} samples (Stratified)")

    # 3. Fit Preprocessing Pipeline
    preprocessor = create_preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # 4. Define Individual Classifiers
    individual_models = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        'Support Vector Machine': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    }

    # 5. Define Primary Hybrid Voting Classifier (Soft Voting)
    hybrid_voting = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)),
            ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42))
        ],
        voting='soft',
        weights=[1.0, 1.5, 1.2, 1.2]
    )

    # 6. Define Secondary Stacking Classifier for comparison
    stacking_clf = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42))
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=3
    )

    all_models = {
        **individual_models,
        'Hybrid Voting Ensemble': hybrid_voting,
        'Stacking Classifier': stacking_clf
    }

    # 7. Model Training & Comprehensive Evaluation
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_results = {}
    roc_curve_data = {}
    fitted_models = {}

    print("\nEvaluating Models with 5-Fold Cross-Validation and Test Set Performance:")
    print("-" * 80)

    for name, model in all_models.items():
        # Fit on training data
        model.fit(X_train_preprocessed, y_train)
        fitted_models[name] = model

        # Test set predictions
        y_pred = model.predict(X_test_preprocessed)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
        else:
            y_prob = model.decision_function(X_test_preprocessed)

        # Standard Metrics
        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred, zero_division=0))
        test_rec = float(recall_score(y_test, y_pred, zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
        test_auc = float(roc_auc_score(y_test, y_prob))

        # 5-Fold Stratified Cross Validation
        cv_scores = cross_validate(
            model,
            X_train_preprocessed,
            y_train,
            cv=cv_kfold,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )

        cv_acc = float(cv_scores['test_accuracy'].mean())
        cv_acc_std = float(cv_scores['test_accuracy'].std())
        cv_f1 = float(cv_scores['test_f1'].mean())
        cv_auc = float(cv_scores['test_roc_auc'].mean())

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # ROC Curve Coordinates
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Downsample ROC points for clean JSON
        roc_step = max(1, len(fpr) // 15)
        roc_curve_data[name] = {
            'fpr': [round(float(val), 4) for val in fpr[::roc_step].tolist()] + [1.0],
            'tpr': [round(float(val), 4) for val in tpr[::roc_step].tolist()] + [1.0]
        }

        metrics_results[name] = {
            'test_accuracy': round(test_acc * 100, 2),
            'test_precision': round(test_prec * 100, 2),
            'test_recall': round(test_rec * 100, 2),
            'test_f1': round(test_f1 * 100, 2),
            'test_auc': round(test_auc * 100, 2),
            'cv_accuracy': round(cv_acc * 100, 2),
            'cv_accuracy_std': round(cv_acc_std * 100, 2),
            'cv_f1': round(cv_f1 * 100, 2),
            'cv_auc': round(cv_auc * 100, 2),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
                'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
            },
            'is_hybrid': 'Hybrid' in name or 'Stacking' in name
        }

        print(f"{name:<24} | Test Acc: {test_acc*100:5.2f}% | F1: {test_f1*100:5.2f}% | AUC: {test_auc*100:5.2f}% | CV Acc: {cv_acc*100:5.2f}% (±{cv_acc_std*100:4.2f}%)")

    # 8. Feature Importance (from Random Forest)
    rf_model = fitted_models['Random Forest']
    # Match feature order: NUMERIC_FEATURES + CATEGORICAL_FEATURES
    all_feature_order = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    importances = rf_model.feature_importances_
    
    feature_importance_list = []
    for feat_name, imp in zip(all_feature_order, importances):
        feature_importance_list.append({
            'feature': feat_name,
            'readable_name': FEATURE_NAMES_READABLE.get(feat_name, feat_name),
            'importance': round(float(imp) * 100, 2)
        })
    feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)

    # 9. Dataset Summary
    dataset_info = {
        'total_samples': total_samples,
        'features_count': features_count,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'positive_samples': pos_count,
        'negative_samples': neg_count,
        'positive_percentage': round((pos_count / total_samples) * 100, 1),
        'negative_percentage': round((neg_count / total_samples) * 100, 1),
        'source': 'UCI Machine Learning Repository - Cleveland Heart Disease Dataset',
        'target_variable': 'Presence of Heart Disease (0: No, 1: Yes)'
    }

    if save_artifacts:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
        os.makedirs(save_dir, exist_ok=True)

        pipeline_path = os.path.join(save_dir, 'trained_pipeline.joblib')
        pipeline_data = {
            'preprocessor': preprocessor,
            'models': {
                'Logistic Regression': fitted_models['Logistic Regression'],
                'Random Forest': fitted_models['Random Forest'],
                'Support Vector Machine': fitted_models['Support Vector Machine'],
                'K-Nearest Neighbors': fitted_models['K-Nearest Neighbors'],
                'Decision Tree': fitted_models['Decision Tree'],
                'Gradient Boosting': fitted_models['Gradient Boosting']
            },
            'hybrid_model': fitted_models['Hybrid Voting Ensemble'],
            'stacking_model': fitted_models['Stacking Classifier'],
            'feature_columns': FEATURE_COLUMNS,
            'feature_names_readable': FEATURE_NAMES_READABLE
        }
        joblib.dump(pipeline_data, pipeline_path)
        print(f"\nSaved model pipeline to: {pipeline_path}")

        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_results, f, indent=2)
        print(f"Saved evaluation metrics to: {metrics_path}")

        dataset_path = os.path.join(save_dir, 'dataset_info.json')
        with open(dataset_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        print(f"Saved dataset metadata to: {dataset_path}")

        roc_path = os.path.join(save_dir, 'roc_curve.json')
        with open(roc_path, 'w') as f:
            json.dump(roc_curve_data, f, indent=2)
        print(f"Saved ROC curve coordinates to: {roc_path}")

        fi_path = os.path.join(save_dir, 'feature_importance.json')
        with open(fi_path, 'w') as f:
            json.dump(feature_importance_list, f, indent=2)
        print(f"Saved feature importances to: {fi_path}")

    print("=" * 60)
    print("Training and Evaluation Completed Successfully!")
    print("=" * 60)
    return metrics_results, dataset_info


if __name__ == '__main__':
    train_and_evaluate()
