"""
Main URL Configuration for Heart Disease Hybrid Machine Learning Platform
"""
from django.contrib import admin
from django.urls import path, include

# Customize Admin Site Headers
admin.site.site_header = "CardioPulse ML Admin"
admin.site.site_title = "CardioPulse Portal"
admin.site.index_title = "Healthcare Analytics & Model Management"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictions.urls')),
    path('accounts/', include('accounts.urls')),
]
