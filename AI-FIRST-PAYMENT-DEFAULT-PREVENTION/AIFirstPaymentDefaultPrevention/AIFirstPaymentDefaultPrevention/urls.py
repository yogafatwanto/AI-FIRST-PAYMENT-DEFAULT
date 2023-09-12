from django.contrib import admin
from django.urls import path, include
from modelbuild import views


urlpatterns = [
    path('', include('modelbuild.urls')),
    path("admin/", admin.site.urls),
    path('predict/', views.call_predict.as_view(), name = 'predict')
]
