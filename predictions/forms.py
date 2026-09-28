from django import forms


class PatientAssessmentForm(forms.Form):
    """
    Form for entering patient clinical features with human-readable choices and validation.
    """
    patient_name = forms.CharField(
        max_length=150,
        required=False,
        initial="Patient Assessment",
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g. John Doe / Patient #1024',
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    age = forms.FloatField(
        min_value=18,
        max_value=100,
        initial=54,
        widget=forms.NumberInput(attrs={
            'placeholder': 'Age (18 - 100)',
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    sex = forms.ChoiceField(
        choices=[(1, 'Male (1)'), (0, 'Female (0)')],
        initial=1,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    cp = forms.ChoiceField(
        label="Chest Pain Type",
        choices=[
            (1, '1: Typical Angina (Exertional chest discomfort relieved by rest)'),
            (2, '2: Atypical Angina (Chest distress with non-typical features)'),
            (3, '3: Non-Anginal Pain (Discomfort unrelated to cardiac ischemia)'),
            (4, '4: Asymptomatic (No overt pain; frequent in silent ischemia)')
        ],
        initial=1,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    trestbps = forms.FloatField(
        label="Resting Blood Pressure (mm Hg)",
        min_value=60,
        max_value=240,
        initial=130,
        widget=forms.NumberInput(attrs={
            'placeholder': 'e.g. 120 - 160 mm Hg',
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    chol = forms.FloatField(
        label="Serum Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        initial=240,
        widget=forms.NumberInput(attrs={
            'placeholder': 'e.g. 150 - 320 mg/dL',
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    fbs = forms.ChoiceField(
        label="Fasting Blood Sugar > 120 mg/dL",
        choices=[
            (0, '0: False (Normal <= 120 mg/dL)'),
            (1, '1: True (Elevated > 120 mg/dL - Diabetic/Impaired)')
        ],
        initial=0,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    restecg = forms.ChoiceField(
        label="Resting Electrocardiogram (ECG)",
        choices=[
            (0, '0: Normal'),
            (1, '1: ST-T Wave Abnormality (T-wave inversion, ST displacement)'),
            (2, '2: Left Ventricular Hypertrophy (LVH by Estes criteria)')
        ],
        initial=0,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    thalach = forms.FloatField(
        label="Maximum Heart Rate Achieved (bpm)",
        min_value=60,
        max_value=230,
        initial=150,
        widget=forms.NumberInput(attrs={
            'placeholder': 'e.g. 110 - 190 bpm',
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    exang = forms.ChoiceField(
        label="Exercise-Induced Angina",
        choices=[
            (0, '0: No (No angina provoked during exercise)'),
            (1, '1: Yes (Angina provoked during stress test)')
        ],
        initial=0,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    oldpeak = forms.FloatField(
        label="ST Depression (Oldpeak in mm)",
        min_value=0.0,
        max_value=7.0,
        initial=1.0,
        widget=forms.NumberInput(attrs={
            'placeholder': 'e.g. 0.0 - 4.5 mm',
            'step': '0.1',
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    slope = forms.ChoiceField(
        label="Slope of Peak Exercise ST Segment",
        choices=[
            (1, '1: Upsloping (Physiologically normal exertion response)'),
            (2, '2: Flat (Ischemic horizontal depression indicator)'),
            (3, '3: Downsloping (Severe ischemia / significant coronary risk)')
        ],
        initial=1,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    ca = forms.ChoiceField(
        label="Number of Major Vessels Colored by Fluoroscopy (0-3)",
        choices=[
            (0, '0 Vessels Colored (No fluoroscopy obstruction)'),
            (1, '1 Vessel Colored'),
            (2, '2 Vessels Colored'),
            (3, '3 Vessels Colored (Multi-vessel involvement)')
        ],
        initial=0,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    thal = forms.ChoiceField(
        label="Thalassemia Perfusion Scintigraphy (Thal)",
        choices=[
            (3, '3: Normal Perfusion (Adequate myocardial uptake)'),
            (6, '6: Fixed Defect (Previous non-reversible infarction)'),
            (7, '7: Reversible Defect (Ischemic transient perfusion loss)')
        ],
        initial=3,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )

    notes = forms.CharField(
        label="Clinical Notes / Observations (Optional)",
        required=False,
        widget=forms.Textarea(attrs={
            'rows': 2,
            'placeholder': 'Add optional notes, referring physician, or clinical context...',
            'class': 'w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-teal-500/50'
        })
    )
