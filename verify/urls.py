from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.upload_view, name='upload_photos'),
      # Doit correspondre au nom de la vuepath
    path('upload-face/', views.upload_view, name='upload_view'),
    path('accept-photo/', views.accept_photo, name='accept_photo'),
    path('compare_face/', include('identity_verification.verify.compare_face.urls')),
]
