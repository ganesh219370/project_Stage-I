"""
Services layer for Prediction operations, dashboard statistics aggregation,
and database syncing with evaluated model metrics.
"""

from django.db.models import Avg, Count, Q
from .models import Prediction, ModelPerformanceMetadata
from ml.evaluate import get_model_metrics


def save_prediction_record(user, patient_name, prediction_result, notes=""):
    """
    Persists an executed prediction into the database linked to the authenticated user.
    """
    patient_data = prediction_result.get('patient_data', {})
    
    record = Prediction.objects.create(
        user=user,
        patient_name=patient_name or 'Patient Record',
        age=patient_data.get('age', 0),
        sex=int(patient_data.get('sex', 0)),
        cp=int(patient_data.get('cp', 1)),
        trestbps=patient_data.get('trestbps', 0),
        chol=patient_data.get('chol', 0),
        fbs=int(patient_data.get('fbs', 0)),
        restecg=int(patient_data.get('restecg', 0)),
        thalach=patient_data.get('thalach', 0),
        exang=int(patient_data.get('exang', 0)),
        oldpeak=patient_data.get('oldpeak', 0.0),
        slope=int(patient_data.get('slope', 1)),
        ca=int(patient_data.get('ca', 0)),
        thal=int(patient_data.get('thal', 3)),
        prediction=prediction_result.get('final_prediction', 0),
        probability=prediction_result.get('final_probability', 0.0),
        confidence=prediction_result.get('final_confidence', 0.0),
        risk_level=prediction_result.get('risk_level', 'Low Risk'),
        model_name=prediction_result.get('model_name', 'Hybrid Voting Ensemble (Soft Voting)'),
        individual_results=prediction_result.get('individual_models', {}),
        clinical_factors=prediction_result.get('clinical_factors', []),
        notes=notes
    )
    return record


def get_user_dashboard_stats(user):
    """
    Aggregates metrics and statistics for user's personal dashboard.
    """
    user_predictions = Prediction.objects.filter(user=user)
    total_count = user_predictions.count()

    if total_count == 0:
        return {
            'total_predictions': 0,
            'high_risk_count': 0,
            'moderate_risk_count': 0,
            'low_risk_count': 0,
            'avg_confidence': 0.0,
            'latest_prediction': None,
            'risk_distribution': {'High Risk': 0, 'Moderate Risk': 0, 'Low Risk': 0},
            'recent_predictions': [],
            'timeline_dates': [],
            'timeline_probs': [],
            'timeline_labels': []
        }

    high_risk_count = user_predictions.filter(prediction=1).count()
    low_risk_count = user_predictions.filter(prediction=0).count()
    
    # Moderate risk (prob between 40 and 70)
    moderate_risk_count = user_predictions.filter(probability__gte=40.0, probability__lt=70.0).count()
    
    avg_conf = user_predictions.aggregate(avg=Avg('confidence'))['avg'] or 0.0
    latest = user_predictions.first()
    recent = user_predictions[:6]

    # Timeline data for charts (last 10 predictions reversed for chronological order)
    timeline_records = list(user_predictions[:10])
    timeline_records.reverse()
    timeline_dates = [r.created_at.strftime('%b %d, %H:%M') for r in timeline_records]
    timeline_probs = [r.probability for r in timeline_records]
    timeline_labels = [r.patient_name for r in timeline_records]

    return {
        'total_predictions': total_count,
        'high_risk_count': high_risk_count,
        'low_risk_count': low_risk_count,
        'moderate_risk_count': moderate_risk_count,
        'avg_confidence': round(avg_conf, 1),
        'latest_prediction': latest,
        'risk_distribution': {
            'High Risk': high_risk_count,
            'Low Risk': low_risk_count,
        },
        'recent_predictions': recent,
        'timeline_dates': timeline_dates,
        'timeline_probs': timeline_probs,
        'timeline_labels': timeline_labels
    }


def sync_model_metadata_in_db():
    """
    Syncs the evaluated machine learning metrics from metrics.json to the database.
    """
    metrics = get_model_metrics()
    for name, m in metrics.items():
        ModelPerformanceMetadata.objects.update_or_create(
            model_name=name,
            defaults={
                'version': 'v1.0-cleveland',
                'accuracy': m.get('test_accuracy', 0.0),
                'precision': m.get('test_precision', 0.0),
                'recall': m.get('test_recall', 0.0),
                'f1_score': m.get('test_f1', 0.0),
                'roc_auc': m.get('test_auc', 0.0),
                'cv_accuracy': m.get('cv_accuracy', 0.0),
                'cv_std': m.get('cv_accuracy_std', 0.0),
                'is_hybrid': m.get('is_hybrid', False)
            }
        )
