from django.urls import path
from . import views
from django.urls import path, include


urlpatterns = [
   path('compare-faces/', views.compare_faces_page, name='compare_faces_page'),
    path('compare-faces/submit/', views.compare_faces_with_images, name='compare_faces_with_images'),


]
