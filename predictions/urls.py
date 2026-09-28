from django.urls import path
from . import views

urlpatterns = [
    # Web UI Pages
    path('', views.home_view, name='home'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
    path('result/<int:prediction_id>/', views.result_view, name='prediction_result'),
    path('history/', views.history_view, name='prediction_history'),
    path('history/<int:prediction_id>/delete/', views.delete_prediction_view, name='delete_prediction'),
    path('history/export/', views.export_history_csv_view, name='export_history_csv'),
    path('performance/', views.performance_view, name='model_performance'),
    path('insights/', views.insights_view, name='model_insights'),

    # REST APIs
    path('api/predict/', views.PredictAPIView.as_view(), name='api_predict'),
    path('api/predictions/', views.PredictionListAPIView.as_view(), name='api_prediction_list'),
    path('api/predictions/<int:pk>/', views.PredictionDetailAPIView.as_view(), name='api_prediction_detail'),
    path('api/model-performance/', views.ModelPerformanceAPIView.as_view(), name='api_model_performance'),
    path('api/dashboard-stats/', views.DashboardStatsAPIView.as_view(), name='api_dashboard_stats'),
]
