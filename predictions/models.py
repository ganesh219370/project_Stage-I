from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):
    """
    Stores individual patient heart disease assessments and ensemble predictions.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    patient_name = models.CharField(max_length=150, blank=True, default='Patient Record')
    created_at = models.DateTimeField(auto_now_add=True)

    # 13 Clinical Attributes
    age = models.FloatField(help_text="Age in years")
    sex = models.IntegerField(help_text="0: Female, 1: Male")
    cp = models.IntegerField(help_text="Chest Pain Type (1-4)")
    trestbps = models.FloatField(help_text="Resting Blood Pressure (mm Hg)")
    chol = models.FloatField(help_text="Serum Cholesterol (mg/dL)")
    fbs = models.IntegerField(help_text="Fasting Blood Sugar > 120 mg/dL (0 or 1)")
    restecg = models.IntegerField(help_text="Resting ECG Results (0, 1, 2)")
    thalach = models.FloatField(help_text="Maximum Heart Rate Achieved (bpm)")
    exang = models.IntegerField(help_text="Exercise-Induced Angina (0: No, 1: Yes)")
    oldpeak = models.FloatField(help_text="ST Depression induced by exercise")
    slope = models.IntegerField(help_text="Slope of peak exercise ST segment (1, 2, 3)")
    ca = models.IntegerField(help_text="Major vessels colored by fluoroscopy (0-3)")
    thal = models.IntegerField(help_text="Thalassemia (3: Normal, 6: Fixed, 7: Reversible)")

    # Prediction Outputs
    prediction = models.IntegerField(help_text="0: No Risk, 1: Heart Disease Detected")
    probability = models.FloatField(help_text="Heart Disease Probability in %")
    confidence = models.FloatField(help_text="Model Confidence in %")
    risk_level = models.CharField(max_length=50, default="Low Risk")
    model_name = models.CharField(max_length=150, default="Hybrid Voting Ensemble (Soft Voting)")

    # Serialized Model Breakdown & Risk Flags
    individual_results = models.JSONField(default=dict, blank=True)
    clinical_factors = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
        ]

    def __str__(self):
        return f"{self.patient_name} ({self.created_at.strftime('%Y-%m-%d %H:%M')}) - {self.risk_level}"

    @property
    def is_disease_detected(self):
        return self.prediction == 1

    @property
    def sex_display(self):
        return "Male" if self.sex == 1 else "Female"

    @property
    def cp_display(self):
        mapping = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'}
        return mapping.get(self.cp, f"Type {self.cp}")

    @property
    def fbs_display(self):
        return "> 120 mg/dL (Elevated)" if self.fbs == 1 else "<= 120 mg/dL (Normal)"

    @property
    def restecg_display(self):
        mapping = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
        return mapping.get(self.restecg, f"Mode {self.restecg}")

    @property
    def exang_display(self):
        return "Yes (Induced)" if self.exang == 1 else "No"

    @property
    def slope_display(self):
        mapping = {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}
        return mapping.get(self.slope, f"Slope {self.slope}")

    @property
    def thal_display(self):
        mapping = {3: 'Normal Perfusion', 6: 'Fixed Defect', 7: 'Reversible Defect'}
        return mapping.get(self.thal, f"Thal {self.thal}")


class ModelPerformanceMetadata(models.Model):
    """
    Stores evaluated machine learning model metadata for admin tracking and reporting.
    """
    model_name = models.CharField(max_length=150, unique=True)
    version = models.CharField(max_length=50, default="v1.0")
    accuracy = models.FloatField(help_text="Test Accuracy %")
    precision = models.FloatField(help_text="Test Precision %")
    recall = models.FloatField(help_text="Test Recall %")
    f1_score = models.FloatField(help_text="Test F1-Score %")
    roc_auc = models.FloatField(help_text="Test ROC-AUC %")
    cv_accuracy = models.FloatField(help_text="5-Fold CV Accuracy %")
    cv_std = models.FloatField(help_text="5-Fold CV Standard Deviation %", default=0.0)
    is_hybrid = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-accuracy']
        verbose_name = 'Model Performance'
        verbose_name_plural = 'Model Performances'

    def __str__(self):
        return f"{self.model_name} (Acc: {self.accuracy}%, AUC: {self.roc_auc}%)"
