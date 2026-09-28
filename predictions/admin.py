from django.contrib import admin
from .models import Prediction, ModelPerformanceMetadata


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('patient_name', 'user', 'risk_level', 'probability', 'confidence', 'created_at')
    list_filter = ('risk_level', 'prediction', 'sex', 'created_at')
    search_fields = ('patient_name', 'user__username', 'user__email')
    readonly_fields = ('created_at', 'probability', 'confidence', 'prediction', 'individual_results', 'clinical_factors')
    ordering = ('-created_at',)


@admin.register(ModelPerformanceMetadata)
class ModelPerformanceMetadataAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'version', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_accuracy', 'is_hybrid', 'updated_at')
    list_filter = ('is_hybrid',)
    search_fields = ('model_name',)
