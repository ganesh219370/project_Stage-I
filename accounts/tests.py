from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse


class AccountsAuthenticationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.existing_user = User.objects.create_user(
            username='cardiologist1',
            email='cardio1@hospital.org',
            password='Password123!',
            first_name='Dr. Smith'
        )

    def test_registration_success(self):
        payload = {
            'first_name': 'Dr. Alice Watson',
            'username': 'alicewatson',
            'email': 'alice@hospital.org',
            'password': 'StrongPassword123!',
            'confirm_password': 'StrongPassword123!',
            'agree_terms': 'on'
        }
        response = self.client.post(reverse('register'), payload)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username='alicewatson').exists())

    def test_registration_password_mismatch(self):
        payload = {
            'first_name': 'Dr. Bob',
            'username': 'bobby',
            'email': 'bob@hospital.org',
            'password': 'Password123!',
            'confirm_password': 'DifferentPassword123!',
            'agree_terms': 'on'
        }
        response = self.client.post(reverse('register'), payload)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='bobby').exists())

    def test_login_with_username_and_email(self):
        # Username login
        response = self.client.post(reverse('login'), {
            'username_or_email': 'cardiologist1',
            'password': 'Password123!'
        })
        self.assertEqual(response.status_code, 302)

        self.client.logout()

        # Email login
        response = self.client.post(reverse('login'), {
            'username_or_email': 'cardio1@hospital.org',
            'password': 'Password123!'
        })
        self.assertEqual(response.status_code, 302)

    def test_logout(self):
        self.client.login(username='cardiologist1', password='Password123!')
        response = self.client.get(reverse('logout'))
        self.assertEqual(response.status_code, 302)
