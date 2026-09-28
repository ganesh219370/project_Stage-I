"""
Evaluation metrics loader and performance analyzer.
Provides metrics, ROC curve data, confusion matrices, and feature importances for views & APIs.
"""

import os
import json


def _load_json(filename):
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'trained_models',
        filename
    )
    if not os.path.exists(filepath):
        # Auto-train if artifacts are missing
        from ml.train import train_and_evaluate
        train_and_evaluate(save_artifacts=True)
    
    with open(filepath, 'r') as f:
        return json.load(f)


def get_model_metrics():
    """Returns evaluation metrics dictionary for all trained models."""
    return _load_json('metrics.json')


def get_dataset_info():
    """Returns dataset summary metadata."""
    return _load_json('dataset_info.json')


def get_roc_curve_data():
    """Returns ROC curve coordinates for Chart.js."""
    return _load_json('roc_curve.json')


def get_feature_importances():
    """Returns sorted feature importances from Random Forest."""
    return _load_json('feature_importance.json')


def get_research_comparison_table():
    """
    Returns structured list of models ordered for research comparison table:
    Individual models first, followed by Hybrid Ensemble and Stacking.
    """
    metrics = get_model_metrics()
    
    table_rows = []
    for model_name, data in metrics.items():
        table_rows.append({
            'model_name': model_name,
            'is_hybrid': data.get('is_hybrid', False),
            'test_accuracy': data.get('test_accuracy'),
            'test_precision': data.get('test_precision'),
            'test_recall': data.get('test_recall'),
            'test_f1': data.get('test_f1'),
            'test_auc': data.get('test_auc'),
            'cv_accuracy': data.get('cv_accuracy'),
            'cv_accuracy_std': data.get('cv_accuracy_std'),
            'cv_auc': data.get('cv_auc'),
            'confusion_matrix': data.get('confusion_matrix')
        })
    
    # Sort: individual models first, hybrid models highlighted
    return table_rows
