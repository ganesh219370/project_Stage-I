import csv
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated, AllowAny

from .models import Prediction, ModelPerformanceMetadata
from .forms import PatientAssessmentForm
from .services import save_prediction_record, get_user_dashboard_stats, sync_model_metadata_in_db
from .serializers import PredictionInputSerializer, PredictionHistorySerializer, ModelPerformanceSerializer
from ml.predict import predict_patient
from ml.evaluate import (
    get_model_metrics,
    get_dataset_info,
    get_roc_curve_data,
    get_feature_importances,
    get_research_comparison_table
)


# ==========================================
# WEB VIEWS
# ==========================================

def home_view(request):
    """
    Public Healthcare Technology Landing Page.
    """
    metrics = get_model_metrics()
    dataset_info = get_dataset_info()
    hybrid_metrics = metrics.get('Hybrid Voting Ensemble', {})

    context = {
        'hybrid_metrics': hybrid_metrics,
        'dataset_info': dataset_info,
    }
    return render(request, 'home.html', context)


@login_required
def dashboard_view(request):
    """
    Authenticated User Dashboard with metrics, charts, and quick actions.
    """
    stats = get_user_dashboard_stats(request.user)
    metrics = get_model_metrics()
    hybrid_metrics = metrics.get('Hybrid Voting Ensemble', {})

    context = {
        'stats': stats,
        'hybrid_metrics': hybrid_metrics,
        'timeline_dates_json': json.dumps(stats['timeline_dates']),
        'timeline_probs_json': json.dumps(stats['timeline_probs']),
        'timeline_labels_json': json.dumps(stats['timeline_labels']),
    }
    return render(request, 'predictions/dashboard.html', context)


@login_required
def predict_view(request):
    """
    Multi-section clinical parameter assessment form.
    Executes ML inference and persists results.
    """
    if request.method == 'POST':
        form = PatientAssessmentForm(request.POST)
        if form.is_valid():
            patient_name = form.cleaned_data.get('patient_name') or 'Patient Assessment'
            notes = form.cleaned_data.get('notes', '')

            # Extract 13 attributes
            patient_params = {
                'age': form.cleaned_data['age'],
                'sex': form.cleaned_data['sex'],
                'cp': form.cleaned_data['cp'],
                'trestbps': form.cleaned_data['trestbps'],
                'chol': form.cleaned_data['chol'],
                'fbs': form.cleaned_data['fbs'],
                'restecg': form.cleaned_data['restecg'],
                'thalach': form.cleaned_data['thalach'],
                'exang': form.cleaned_data['exang'],
                'oldpeak': form.cleaned_data['oldpeak'],
                'slope': form.cleaned_data['slope'],
                'ca': form.cleaned_data['ca'],
                'thal': form.cleaned_data['thal'],
            }

            result = predict_patient(patient_params)

            if result.get('success'):
                # Save to database
                record = save_prediction_record(
                    user=request.user,
                    patient_name=patient_name,
                    prediction_result=result,
                    notes=notes
                )
                messages.success(request, f"Clinical analysis completed for {record.patient_name}.")
                return redirect('prediction_result', prediction_id=record.id)
            else:
                for field, err in result.get('errors', {}).items():
                    messages.error(request, f"{field}: {err}")
        else:
            messages.error(request, "Please check the form inputs. Some fields have invalid values.")
    else:
        form = PatientAssessmentForm()

    return render(request, 'predictions/predict_form.html', {'form': form})


@login_required
def result_view(request, prediction_id):
    """
    Displays comprehensive prediction results, individual model breakdown,
    probability gauge, risk explanation, and disclaimer.
    """
    record = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    
    # Model comparison bar data for chart
    model_names = list(record.individual_results.keys()) + ['Hybrid Voting Model']
    model_probs = [v['probability'] for v in record.individual_results.values()] + [record.probability]

    context = {
        'record': record,
        'model_names_json': json.dumps(model_names),
        'model_probs_json': json.dumps(model_probs),
    }
    return render(request, 'predictions/result.html', context)


@login_required
def history_view(request):
    """
    Prediction history table with search, risk filter, sorting, and pagination.
    """
    query = request.GET.get('q', '').strip()
    risk_filter = request.GET.get('risk', '')
    sort_by = request.GET.get('sort', '-created_at')

    queryset = Prediction.objects.filter(user=request.user)

    if query:
        queryset = queryset.filter(
            Q(patient_name__icontains=query) | Q(notes__icontains=query)
        )

    if risk_filter == 'high':
        queryset = queryset.filter(prediction=1)
    elif risk_filter == 'low':
        queryset = queryset.filter(prediction=0)

    # Allowed sort fields
    allowed_sorts = ['-created_at', 'created_at', '-probability', 'probability', 'patient_name', '-patient_name']
    if sort_by in allowed_sorts:
        queryset = queryset.order_by(sort_by)
    else:
        queryset = queryset.order_by('-created_at')

    paginator = Paginator(queryset, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'query': query,
        'risk_filter': risk_filter,
        'sort_by': sort_by,
        'total_count': queryset.count(),
    }
    return render(request, 'predictions/history.html', context)


@login_required
def delete_prediction_view(request, prediction_id):
    """
    Deletes an individual prediction record.
    """
    record = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    patient_name = record.patient_name
    record.delete()
    messages.success(request, f"Assessment for '{patient_name}' has been deleted from history.")
    return redirect('prediction_history')


@login_required
def export_history_csv_view(request):
    """
    Exports user prediction history as CSV.
    """
    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="heart_disease_predictions.csv"'
    response.write('\ufeff')

    writer = csv.writer(response)
    writer.writerow([
        'ID', 'Date', 'Patient Name', 'Age', 'Sex', 'Chest Pain Type',
        'Resting BP', 'Cholesterol', 'Fasting Blood Sugar', 'Resting ECG',
        'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'ST Slope',
        'Major Vessels', 'Thalassemia', 'Prediction', 'Probability (%)',
        'Confidence (%)', 'Risk Level', 'Model Name'
    ])

    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    for p in predictions:
        writer.writerow([
            p.id,
            p.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            p.patient_name,
            p.age,
            p.sex_display,
            p.cp_display,
            p.trestbps,
            p.chol,
            p.fbs_display,
            p.restecg_display,
            p.thalach,
            p.exang_display,
            p.oldpeak,
            p.slope_display,
            p.ca,
            p.thal_display,
            p.prediction,
            p.probability,
            p.confidence,
            p.risk_level,
            p.model_name
        ])

    return response


def performance_view(request):
    """
    Dedicated Model Performance page with interactive Chart.js comparison charts,
    confusion matrices, and cross-validation summaries.
    """
    sync_model_metadata_in_db()
    comparison_table = get_research_comparison_table()
    metrics = get_model_metrics()

    # Prepare chart labels & values
    model_labels = [row['model_name'] for row in comparison_table]
    acc_scores = [row['test_accuracy'] for row in comparison_table]
    prec_scores = [row['test_precision'] for row in comparison_table]
    rec_scores = [row['test_recall'] for row in comparison_table]
    f1_scores = [row['test_f1'] for row in comparison_table]
    auc_scores = [row['test_auc'] for row in comparison_table]
    cv_scores = [row['cv_accuracy'] for row in comparison_table]

    hybrid_row = next((r for r in comparison_table if 'Hybrid' in r['model_name']), comparison_table[0])

    context = {
        'comparison_table': comparison_table,
        'hybrid_row': hybrid_row,
        'model_labels_json': json.dumps(model_labels),
        'acc_scores_json': json.dumps(acc_scores),
        'prec_scores_json': json.dumps(prec_scores),
        'rec_scores_json': json.dumps(rec_scores),
        'f1_scores_json': json.dumps(f1_scores),
        'auc_scores_json': json.dumps(auc_scores),
        'cv_scores_json': json.dumps(cv_scores),
    }
    return render(request, 'predictions/performance.html', context)


def insights_view(request):
    """
    Model Insights & Academic Research Comparison page.
    Displays dataset summary, ROC curves, feature importances, and academic writeup.
    """
    dataset_info = get_dataset_info()
    roc_data = get_roc_curve_data()
    feature_importances = get_feature_importances()
    metrics = get_model_metrics()

    # Feature importance chart labels and values
    fi_labels = [item['readable_name'] for item in feature_importances]
    fi_values = [item['importance'] for item in feature_importances]

    context = {
        'dataset_info': dataset_info,
        'feature_importances': feature_importances,
        'roc_data_json': json.dumps(roc_data),
        'fi_labels_json': json.dumps(fi_labels),
        'fi_values_json': json.dumps(fi_values),
        'metrics': metrics
    }
    return render(request, 'predictions/insights.html', context)


# ==========================================
# REST API ENDPOINTS
# ==========================================

class PredictAPIView(APIView):
    """
    POST /api/predict/
    Accepts patient parameters, executes ML prediction, and optionally saves to history.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({'success': False, 'errors': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data
        save_to_history = data.pop('save_to_history', True)
        patient_name = data.pop('patient_name', 'Patient Assessment')
        notes = data.pop('notes', '')

        result = predict_patient(data)

        if not result.get('success'):
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        saved_id = None
        if save_to_history:
            record = save_prediction_record(
                user=request.user,
                patient_name=patient_name,
                prediction_result=result,
                notes=notes
            )
            saved_id = record.id

        result['saved_id'] = saved_id
        return Response(result, status=status.HTTP_200_OK)


class PredictionListAPIView(APIView):
    """
    GET /api/predictions/
    Returns list of prediction history for the authenticated user.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        predictions = Prediction.objects.filter(user=request.user)
        serializer = PredictionHistorySerializer(predictions, many=True)
        return Response({'success': True, 'count': predictions.count(), 'results': serializer.data})


class PredictionDetailAPIView(APIView):
    """
    GET /api/predictions/<id>/
    DELETE /api/predictions/<id>/
    """
    permission_classes = [IsAuthenticated]

    def get_object(self, pk, user):
        return get_object_or_404(Prediction, id=pk, user=user)

    def get(self, request, pk):
        prediction = self.get_object(pk, request.user)
        serializer = PredictionHistorySerializer(prediction)
        return Response({'success': True, 'prediction': serializer.data})

    def delete(self, request, pk):
        prediction = self.get_object(pk, request.user)
        prediction.delete()
        return Response({'success': True, 'message': 'Prediction record deleted successfully.'}, status=status.HTTP_204_NO_CONTENT)


class ModelPerformanceAPIView(APIView):
    """
    GET /api/model-performance/
    Returns evaluated model metrics, ROC coordinates, and research comparison table.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        metrics = get_model_metrics()
        table = get_research_comparison_table()
        roc = get_roc_curve_data()
        dataset = get_dataset_info()

        return Response({
            'success': True,
            'dataset': dataset,
            'comparison_table': table,
            'metrics': metrics,
            'roc_curve': roc
        })


class DashboardStatsAPIView(APIView):
    """
    GET /api/dashboard-stats/
    Returns authenticated user's dashboard statistics.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        stats = get_user_dashboard_stats(request.user)
        # Remove non-serializable objects for JSON API
        stats_clean = {
            'total_predictions': stats['total_predictions'],
            'high_risk_count': stats['high_risk_count'],
            'low_risk_count': stats['low_risk_count'],
            'moderate_risk_count': stats['moderate_risk_count'],
            'avg_confidence': stats['avg_confidence'],
            'timeline_dates': stats['timeline_dates'],
            'timeline_probs': stats['timeline_probs'],
            'timeline_labels': stats['timeline_labels'],
        }
        return Response({'success': True, 'stats': stats_clean})
