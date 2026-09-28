from rest_framework import serializers
from .models import Prediction, ModelPerformanceMetadata


class PredictionInputSerializer(serializers.Serializer):
    patient_name = serializers.CharField(max_length=150, required=False, default="Patient Assessment")
    age = serializers.FloatField(min_value=18, max_value=100)
    sex = serializers.ChoiceField(choices=[(0, 'Female'), (1, 'Male')])
    cp = serializers.ChoiceField(choices=[(1, 'Typical Angina'), (2, 'Atypical Angina'), (3, 'Non-anginal'), (4, 'Asymptomatic')])
    trestbps = serializers.FloatField(min_value=60, max_value=240)
    chol = serializers.FloatField(min_value=100, max_value=600)
    fbs = serializers.ChoiceField(choices=[(0, 'False'), (1, 'True')])
    restecg = serializers.ChoiceField(choices=[(0, 'Normal'), (1, 'ST-T Abnormality'), (2, 'LV Hypertrophy')])
    thalach = serializers.FloatField(min_value=60, max_value=230)
    exang = serializers.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    oldpeak = serializers.FloatField(min_value=0.0, max_value=7.0)
    slope = serializers.ChoiceField(choices=[(1, 'Upsloping'), (2, 'Flat'), (3, 'Downsloping')])
    ca = serializers.ChoiceField(choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3')])
    thal = serializers.ChoiceField(choices=[(3, 'Normal'), (6, 'Fixed Defect'), (7, 'Reversible Defect')])
    notes = serializers.CharField(required=False, allow_blank=True, default="")
    save_to_history = serializers.BooleanField(required=False, default=True)


class PredictionHistorySerializer(serializers.ModelSerializer):
    sex_display = serializers.ReadOnlyField()
    cp_display = serializers.ReadOnlyField()
    fbs_display = serializers.ReadOnlyField()
    restecg_display = serializers.ReadOnlyField()
    exang_display = serializers.ReadOnlyField()
    slope_display = serializers.ReadOnlyField()
    thal_display = serializers.ReadOnlyField()

    class Meta:
        model = Prediction
        fields = [
            'id', 'patient_name', 'created_at', 'age', 'sex', 'sex_display',
            'cp', 'cp_display', 'trestbps', 'chol', 'fbs', 'fbs_display',
            'restecg', 'restecg_display', 'thalach', 'exang', 'exang_display',
            'oldpeak', 'slope', 'slope_display', 'ca', 'thal', 'thal_display',
            'prediction', 'probability', 'confidence', 'risk_level',
            'model_name', 'individual_results', 'clinical_factors', 'notes'
        ]
        read_only_fields = ['id', 'created_at', 'prediction', 'probability', 'confidence', 'risk_level']


class ModelPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelPerformanceMetadata
        fields = '__all__'
