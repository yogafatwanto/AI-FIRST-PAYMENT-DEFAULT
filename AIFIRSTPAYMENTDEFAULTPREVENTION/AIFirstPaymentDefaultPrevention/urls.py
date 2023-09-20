from django.contrib import admin
from django.urls import path, include
from modelbuild import views


urlpatterns = [
    path('', include('modelbuild.urls')),
    path("admin/", admin.site.urls),
    
]
