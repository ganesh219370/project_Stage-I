from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from predictions.models import Prediction, ModelPerformanceMetadata
from predictions.services import get_user_dashboard_stats
from ml.predict import predict_patient
from ml.evaluate import get_model_metrics, get_dataset_info


class MachineLearningPipelineTests(TestCase):
    def test_ml_pipeline_loads_and_predicts(self):
        """Test that ML pipeline correctly evaluates a high risk and a low risk case."""
        # High-risk profile
        high_risk_input = {
            'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
            'fbs': 0, 'restecg': 2, 'thalach': 108, 'exang': 1, 'oldpeak': 2.6,
            'slope': 2, 'ca': 3, 'thal': 7
        }
        res_high = predict_patient(high_risk_input)
        self.assertTrue(res_high['success'])
        self.assertEqual(res_high['final_prediction'], 1)
        self.assertGreaterEqual(res_high['final_probability'], 50.0)
        self.assertIn('Random Forest', res_high['individual_models'])
        self.assertIn('Logistic Regression', res_high['individual_models'])

        # Low-risk profile
        low_risk_input = {
            'age': 35, 'sex': 0, 'cp': 2, 'trestbps': 115, 'chol': 180,
            'fbs': 0, 'restecg': 0, 'thalach': 175, 'exang': 0, 'oldpeak': 0.1,
            'slope': 1, 'ca': 0, 'thal': 3
        }
        res_low = predict_patient(low_risk_input)
        self.assertTrue(res_low['success'])
        self.assertEqual(res_low['final_prediction'], 0)
        self.assertLess(res_low['final_probability'], 50.0)

    def test_ml_evaluation_metrics_present(self):
        """Test that actual computed metrics are loaded."""
        metrics = get_model_metrics()
        self.assertIn('Hybrid Voting Ensemble', metrics)
        self.assertIn('Random Forest', metrics)
        self.assertGreater(metrics['Hybrid Voting Ensemble']['test_accuracy'], 75.0)

        info = get_dataset_info()
        self.assertEqual(info['total_samples'], 303)
        self.assertEqual(info['features_count'], 13)


class PredictionViewsAndFlowTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='test_dr',
            email='dr@test.com',
            password='testpassword123',
            first_name='Dr. Test'
        )
        self.other_user = User.objects.create_user(
            username='other_dr',
            email='other@test.com',
            password='testpassword123',
            first_name='Dr. Other'
        )

    def test_public_views(self):
        """Test public home, performance, and insights pages load with 200 OK."""
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Intelligent Heart Disease")

        response = self.client.get(reverse('model_performance'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research Benchmark Table")

        response = self.client.get(reverse('model_insights'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Why Hybrid Machine Learning")

    def test_authenticated_dashboard_and_prediction(self):
        """Test login, dashboard access, prediction submission, and result display."""
        # Login
        logged_in = self.client.login(username='test_dr', password='testpassword123')
        self.assertTrue(logged_in)

        # Access dashboard
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dr. Test")

        # Submit new prediction form
        post_data = {
            'patient_name': 'Test Subject A',
            'age': 55,
            'sex': 1,
            'cp': 4,
            'trestbps': 140,
            'chol': 250,
            'fbs': 0,
            'restecg': 1,
            'thalach': 130,
            'exang': 1,
            'oldpeak': 1.8,
            'slope': 2,
            'ca': 1,
            'thal': 7,
            'notes': 'Cardiology assessment test'
        }
        response = self.client.post(reverse('predict'), post_data)
        # Should redirect to prediction_result
        self.assertEqual(response.status_code, 302)
        
        # Verify prediction created in DB
        prediction = Prediction.objects.filter(user=self.user, patient_name='Test Subject A').first()
        self.assertIsNotNone(prediction)
        
        # Access result page
        res_page = self.client.get(reverse('prediction_result', kwargs={'prediction_id': prediction.id}))
        self.assertEqual(res_page.status_code, 200)
        self.assertContains(res_page, "Test Subject A")

    def test_user_data_isolation(self):
        """Verify users cannot see or delete another user's prediction records."""
        # Create prediction for other_user
        p_other = Prediction.objects.create(
            user=self.other_user,
            patient_name='Private Patient of Other Dr',
            age=50, sex=1, cp=1, trestbps=120, chol=200, fbs=0, restecg=0,
            thalach=150, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3,
            prediction=0, probability=10.0, confidence=90.0
        )

        # Login as test_dr
        self.client.login(username='test_dr', password='testpassword123')

        # Try accessing other user's prediction result -> should return 404
        response = self.client.get(reverse('prediction_result', kwargs={'prediction_id': p_other.id}))
        self.assertEqual(response.status_code, 404)

        # History page should NOT contain other user's record
        hist_response = self.client.get(reverse('prediction_history'))
        self.assertNotContains(hist_response, "Private Patient of Other Dr")


class RestApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='api_user',
            email='api@test.com',
            password='testpassword123'
        )
        self.client.login(username='api_user', password='testpassword123')

    def test_api_predict_endpoint(self):
        payload = {
            'patient_name': 'API Patient 1',
            'age': 60,
            'sex': 1,
            'cp': 3,
            'trestbps': 135,
            'chol': 220,
            'fbs': 0,
            'restecg': 0,
            'thalach': 145,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 1,
            'ca': 0,
            'thal': 3,
            'save_to_history': True
        }
        response = self.client.post(reverse('api_predict'), data=payload, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('final_probability', data)
        self.assertIn('individual_models', data)
        self.assertIsNotNone(data['saved_id'])

    def test_api_performance_endpoint(self):
        response = self.client.get(reverse('api_model_performance'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('comparison_table', data)
