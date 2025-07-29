import pymongo
from pathlib import Path
import os

# MONGO settings
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "identity_verification"
MONGO_COLLECTION = "users"

# Base directory
BASE_DIR = Path("/home/oumayma/identity_verification")

# Quick-start development settings
SECRET_KEY = 'django-insecure-i=j+2fb1*r#rrbu+^+ug8)ec%o9z&!+2zf9_b0=3p%r_rw#q(j'
DEBUG = True
ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'verify',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'identity_verification.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, 'verify', 'templates'),
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'identity_verification.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'identity_verification_db',
        'HOST': 'localhost',
        'PORT': 27017,
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'verify/static'),  # Points to verify/static/
]

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CompreFace settings
COMPAREFACE = {
    'API_URL': 'http://compreface-api:8000',
    'UI_PUBLIC_URL': 'http://localhost:8001',
    'API_KEYS': {
        'DETECTION': '736500b9-d5c8-482f-b06c-3eb49cdefcf9',
        'RECOGNITION': 'e43199c2-f296-44e6-8143-91b2a1b5077a'
    }
}

