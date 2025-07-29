import os
import logging
import cv2
import base64
import requests
import dlib
import numpy as np
import json
from scipy.spatial import distance
from requests.exceptions import RequestException
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect, ensure_csrf_cookie
from django.core.files.storage import default_storage
from functools import wraps
from django.conf import settings
from pymongo import MongoClient
from bson import Binary

logger = logging.getLogger(__name__)

# Initialize MongoDB client
mongo_client = MongoClient("mongodb://admin:admin@localhost:27017/")
mongo_db = mongo_client["identity_verification_db"]
mongo_collection = mongo_db["identityinfo"]

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

CONFIG = {
    "server_url": "http://localhost:8000",
    "api_keys": {
        "detection": "3b570619-a3c5-4d46-bb78-a1e7d3d0dfe6",
        "recognition": "e4f8cbfa-3585-4eb8-a6ef-5e068d083c20",
        "verification": "05476a69-db31-4539-88b1-d2e8bbf4dcba"
    },
    "endpoints": {
        "detection": "/api/v1/detection/detect",
        "recognition": "/api/v1/recognition/recognize",
        "verification": "/api/v1/verification/verify"
    },
    "verification_threshold": 0.88
}

# Initialize dlib face detector and landmark predictor with correct path
DLIB_DETECTOR = dlib.get_frontal_face_detector()
DLIB_MODEL_PATH = os.path.join(settings.BASE_DIR, 'verify', 'static', 'models', 'shape_predictor_68_face_landmarks.dat')

try:
    DLIB_PREDICTOR = dlib.shape_predictor(DLIB_MODEL_PATH)
    logger.info(f"Successfully loaded dlib shape predictor from: {DLIB_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load dlib shape predictor from {DLIB_MODEL_PATH}: {str(e)}")
    DLIB_PREDICTOR = None

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio (EAR) for blink detection"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def draw_bounding_box(image, rect, color=(0, 255, 0), thickness=2):
    """Draw a rectangle bounding box on the image"""
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

def draw_landmark_points(image, points, color=(0, 0, 255), radius=2):
    """Draw circles on landmark points"""
    for (x, y) in points:
        cv2.circle(image, (x, y), radius, color, -1)

def detect_blinks(image_path, EYE_AR_THRESH=0.25):
    """Detect if eyes are closed using facial landmarks and draw bounding boxes"""
    if DLIB_PREDICTOR is None:
        logger.error("DLIB predictor not initialized - cannot detect blinks")
        return False, None

    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image at {image_path}")
            return False, None
            
        image = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = DLIB_DETECTOR(gray, 0)
        
        if len(rects) == 0:
            logger.info("No faces detected in blink detection")
            return False, image
        
        for rect in rects:
            shape = DLIB_PREDICTOR(gray, rect)
            shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            left_eye = shape_np[42:48]
            right_eye = shape_np[36:42]
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Draw face bounding box
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            draw_bounding_box(image, (x, y, w, h), color=(0, 255, 0))
            
            # Draw eye landmarks
            draw_landmark_points(image, left_eye, color=(255, 0, 0))
            draw_landmark_points(image, right_eye, color=(255, 0, 0))
            
            # Simple debounce: only consider eyes closed if EAR below threshold for at least 2 consecutive frames
            if not hasattr(detect_blinks, "prev_ear"):
                detect_blinks.prev_ear = ear
                detect_blinks.closed_frames = 0
            if ear < EYE_AR_THRESH:
                detect_blinks.closed_frames += 1
            else:
                detect_blinks.closed_frames = 0
            detect_blinks.prev_ear = ear
            
            return detect_blinks.closed_frames >= 2, image
            
    except Exception as e:
        logger.error(f"Error in blink detection: {str(e)}")
        return False, None
        
    return False, image

def perform_liveness_check(image_path):
    """Enhanced liveness check combining traditional and blink detection with bounding boxes"""
    # Traditional face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    img = cv2.imread(image_path)
    if img is None:
        return False, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adjusted scaleFactor and minNeighbors for better detection tolerance
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2)
    
    if len(faces) == 0:
        logger.info("No faces detected in traditional liveness check")
        return False, img
        
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=2)
        if len(eyes) >= 2:
            # Draw bounding boxes on face and eyes
            draw_bounding_box(img, (x, y, w, h), color=(0, 255, 0))
            for (ex, ey, ew, eh) in eyes:
                draw_bounding_box(img, (x+ex, y+ey, ew, eh), color=(255, 0, 0))
            return True, img
    
    # Fall back to blink detection if traditional method fails
    blinked, img_with_blinks = detect_blinks(image_path)
    return blinked, img_with_blinks

def handle_exceptions(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        try:
            return view_func(request, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {view_func.__name__}: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': 'Internal server error'
            }, status=500)
    return wrapper

def encode_image(image_path):
    """Convert image to base64 with validation"""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            if not image_data:
                logger.error("Error: Empty image file")
                return None
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error: {str(e)}")
        return None

def make_api_request(service, payload):
    """Make API request with timeout and error handling"""
    try:
        url = f"{CONFIG['server_url']}{CONFIG['endpoints'][service]}"
        headers = {
            "x-api-key": CONFIG['api_keys'][service],
            "Content-Type": "application/json"
        }
        logger.debug(f"Making API request to {url} with payload keys: {list(payload.keys())}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        logger.debug(f"API response status: {response.status_code}")
        return response.json()
    except RequestException as e:
        logger.error(f"API Error ({service}): {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
            except Exception:
                logger.error(f"Raw response: {e.response.text}")
        return None

@require_http_methods(["GET"])
def compare_faces_page(request):
    return render(request, 'verify/compare_faces.html')

@require_http_methods(["POST"])
@csrf_protect
@ensure_csrf_cookie
@handle_exceptions
def compare_faces_with_images(request):
    source_dir = os.path.join(settings.BASE_DIR, 'media', 'temp')

    try:
        files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        if not files:
            logger.error("No source images found for comparison")
            return JsonResponse({'status': 'error', 'message': 'No source images found for comparison'}, status=400)
        latest_source_path = max(files, key=os.path.getmtime)
        logger.info(f"Latest source image path: {latest_source_path}")
    except Exception as e:
        logger.error(f"Error accessing source images: {str(e)}")
        return JsonResponse({'status': 'error', 'message': f'Error accessing source images: {str(e)}'}, status=500)

    target_file = request.FILES.get('target_image')
    if not target_file:
        logger.error("Target image is required but not received")
        return JsonResponse({'status': 'error', 'message': 'Target image is required'}, status=400)

    # Manual validation of uploaded file
    ext = target_file.name.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.error(f"Unsupported file extension: {ext}")
        return JsonResponse({'status': 'error', 'message': 'Unsupported file extension'}, status=400)
    if target_file.size > MAX_FILE_SIZE:
        logger.error(f"File size exceeds limit: {target_file.size}")
        return JsonResponse({'status': 'error', 'message': 'File size exceeds limit'}, status=400)

    try:
        target_path = default_storage.save('temp_target', target_file)
        logger.info(f"Target image saved at: {target_path}")

        # Perform enhanced liveness check on target image
        liveness_passed, img_with_boxes = perform_liveness_check(default_storage.path(target_path))
        if not bool(liveness_passed):
            logger.error("Liveness check failed on target image")
            return JsonResponse({'status': 'error', 'message': 'Liveness check failed'}, status=400)

        # Perform blink detection to get eyes_closed and bounding boxes
        eyes_closed, img_with_blinks = detect_blinks(default_storage.path(target_path))
        
        blink_count = 0  # Backend does not accumulate blink count per frame

        # Save image with bounding boxes for debugging
        debug_img_path = default_storage.path('temp_debug.jpg')
        cv2.imwrite(debug_img_path, img_with_blinks)

        # Encode target image with bounding boxes
        target_image_b64 = encode_image(debug_img_path)
        

        # Prepare bounding box data (for simplicity, only face bounding box of first face)
        bounding_boxes = []
        if DLIB_PREDICTOR is not None:
            image = cv2.imread(default_storage.path(target_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = DLIB_DETECTOR(gray, 0)
            for rect in rects:
                bounding_boxes.append({
                    'left': rect.left(),
                    'top': rect.top(),
                    'right': rect.right(),
                    'bottom': rect.bottom()
                })

        # Call verification API to compare faces
        verification_payload = {
            "source_image": encode_image(latest_source_path),
            "target_image": target_image_b64
        }
        logger.debug(f"Verification payload keys: {list(verification_payload.keys())}")
        verification_response = make_api_request("verification", verification_payload)
        verified = False
        confidence = 0.0
        message = ""
        if verification_response and verification_response.get("result"):
            similarity = verification_response["result"][0]["face_matches"][0].get("similarity", 0.0)
            verified = similarity >= 0.88
            confidence = round(similarity, 2)  # Confidence as decimal (e.g., 0.65)
            message = "Faces match" if verified else "Faces do not match"
        else:
            logger.error("Verification API call failed or returned error")
            verified = False
            confidence = 0.0
            message = "Verification failed"

        # Save live photo in MongoDB only if verified
        if verified:
            try:
                # Find the most recent document in identityinfo collection
                recent_doc = mongo_collection.find_one(sort=[('_id', -1)])
                if recent_doc:
                    with open(debug_img_path, "rb") as image_file:
                        image_data = image_file.read()
                    result = mongo_collection.update_one(
                        {'_id': recent_doc['_id']},
                        {'$set': {'photo': Binary(image_data)}}
                    )
                    logger.info(f"MongoDB document updated successfully for _id: {recent_doc['_id']}")
                else:
                    logger.warning("No recent document found in MongoDB to update with live photo.")
            except Exception as e:
                logger.error(f"MongoDB update error for live photo: {e}")
            
        # Return liveness, eyes_closed, blink count, bounding boxes, verification result, confidence, and message
        return JsonResponse({
            'status': 'success',
            'data': {
                'liveness_passed': bool(liveness_passed),
                'eyes_closed': bool(eyes_closed),
                'blink_count': blink_count,
                'bounding_boxes': bounding_boxes,
                'verified': verified,
                'confidence': confidence,
                'message': message,
                'target_image_with_boxes': target_image_b64
            },
            'liveness_detection': {
                'liveness_passed': bool(liveness_passed)
            }
        })


    except Exception as e:
        logger.exception(f"Error in compare_faces_with_images: {str(e)}")
        return JsonResponse({'status': 'error', 'message': 'Internal server error'}, status=500)
    finally:
        if 'target_path' in locals():
            default_storage.delete(target_path)
            logger.info(f"Temporary target image deleted: {target_path}")
