import cv2
import base64
import requests
import os
import json
import time
import numpy as np
from requests.exceptions import RequestException

# Configuration
CONFIG = {
    # ... (keep your existing config) ...
    "liveness": {
        "ear_threshold": 0.25,  # Eye aspect ratio threshold
        "consecutive_frames": 3, # Frames needed to register blink
        "check_seconds": 5,      # Max duration for liveness check
        "required_blinks": 3     # Minimum blinks required
    }
}

def eye_aspect_ratio(eye):
    # Calculate eye aspect ratio (simplified without scipy)
    # eye should contain 6 (x,y) coordinates
    width = np.linalg.norm(eye[0] - eye[3])
    height1 = np.linalg.norm(eye[1] - eye[5])
    height2 = np.linalg.norm(eye[2] - eye[4])
    return (height1 + height2) / (2.0 * width)

def detect_blinks(frame, eye_cascade, face_roi):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[face_roi[1]:face_roi[1]+face_roi[3], face_roi[0]:face_roi[0]+face_roi[2]]
    
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
    if len(eyes) >= 2:
        # Simple approximation of eye landmarks
        left_eye = eyes[0]
        right_eye = eyes[1]
        
        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio([
            (left_eye[0], left_eye[1]),
            (left_eye[0], left_eye[1] + left_eye[3]//3),
            (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2),
            (left_eye[0] + left_eye[2], left_eye[1]),
            (left_eye[0] + left_eye[2], left_eye[1] + left_eye[3]//3),
            (left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2)
        ])
        
        right_ear = eye_aspect_ratio([
            (right_eye[0], right_eye[1]),
            (right_eye[0], right_eye[1] + right_eye[3]//3),
            (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2),
            (right_eye[0] + right_eye[2], right_eye[1]),
            (right_eye[0] + right_eye[2], right_eye[1] + right_eye[3]//3),
            (right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2)
        ])
        
        ear = (left_ear + right_ear) / 2.0
        return ear < CONFIG["liveness"]["ear_threshold"]
    return False

def perform_liveness_check(cap):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    blink_counter = 0
    consecutive_frames = 0
    start_time = time.time()
    
    print("Liveness check: Please blink naturally")
    
    while (time.time() - start_time) < CONFIG["liveness"]["check_seconds"]:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            
            if detect_blinks(frame, eye_cascade, (x,y,w,h)):
                consecutive_frames += 1
                if consecutive_frames >= CONFIG["liveness"]["consecutive_frames"]:
                    blink_counter += 1
                    consecutive_frames = 0
            else:
                consecutive_frames = 0
            
            # Display status
            cv2.putText(frame, f"Blinks: {blink_counter}/{CONFIG['liveness']['required_blinks']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Time: {int(CONFIG['liveness']['check_seconds'] - (time.time() - start_time))}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow("Liveness Check", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if blink_counter >= CONFIG["liveness"]["required_blinks"]:
            cv2.destroyWindow("Liveness Check")
            return True
    
    cv2.destroyWindow("Liveness Check")
    return False

def capture_face(cap, image_type, perform_liveness=False):
    """Modified capture function with robust liveness check"""
    if perform_liveness:
        if not perform_liveness_check(cap):
            print("Liveness verification failed - no blink detected!")
            return None
        ret, frame = cap.read()
        if not ret:
            return None
    else:
        ret, frame = cap.read()
        if not ret:
            return None
    
    # Rest of your existing capture_face code...
    # ... (keep your image saving and API calls) ...
