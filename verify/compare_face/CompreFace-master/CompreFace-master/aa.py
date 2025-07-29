import cv2
import base64
import requests
import os
import json
import time
import numpy as np
from requests.exceptions import RequestException
from scipy.spatial import distance as dist

# Configuration - Updated with liveness parameters
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
    "image_paths": {
        "source": "captured_face.jpg",
        "target": "captured_target_face.jpg"
    },
    "verification_threshold": 0.7,  # 70% similarity threshold
    "ear_threshold": 0.25,         # Eye aspect ratio threshold
    "blink_frames": 3,             # Consecutive frames for blink detection
    "liveness_duration": 3         # Seconds for liveness test
}

# Eye landmarks indices (approximate positions)
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

def eye_aspect_ratio(eye):
    """Calculate eye aspect ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_blinks(frame, ear_threshold):
    """Simple blink detection using facial landmarks"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Approximate eye positions (simplified without landmarks)
        eye_region_width = w // 3
        eye_region_height = h // 6
        
        # Left eye ROI
        left_eye_x = x + w // 4
        left_eye_y = y + h // 4
        left_eye_roi = frame[left_eye_y:left_eye_y+eye_region_height, 
                            left_eye_x:left_eye_x+eye_region_width]
        
        # Right eye ROI
        right_eye_x = x + w // 2
        right_eye_y = y + h // 4
        right_eye_roi = frame[right_eye_y:right_eye_y+eye_region_height, 
                             right_eye_x:right_eye_x+eye_region_width]
        
        # Simple blink detection (dark pixel ratio)
        def get_eye_state(eye_roi):
            _, threshold = cv2.threshold(cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY_INV)
            height, width = threshold.shape
            return np.sum(threshold) / (height * width * 255)
        
        left_ratio = get_eye_state(left_eye_roi)
        right_ratio = get_eye_state(right_eye_roi)
        
        avg_ratio = (left_ratio + right_ratio) / 2
        is_blinking = avg_ratio > 0.35  # Threshold for closed eyes
        
        # Draw eye regions for visualization
        cv2.rectangle(frame, (left_eye_x, left_eye_y), 
                     (left_eye_x+eye_region_width, left_eye_y+eye_region_height), 
                     (0, 255, 0), 1)
        cv2.rectangle(frame, (right_eye_x, right_eye_y), 
                     (right_eye_x+eye_region_width, right_eye_y+eye_region_height), 
                     (0, 255, 0), 1)
        
        return frame, 0.3 if is_blinking else 0.4, int(is_blinking), "Blinking" if is_blinking else "Open"
    
    return frame, 0, 0, "No Face"

def perform_liveness_test(cap):
    """Perform liveness test by detecting blinks"""
    print("\nStarting liveness test - please blink naturally...")
    start_time = time.time()
    blink_count = 0
    
    while (time.time() - start_time) < CONFIG['liveness_duration']:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, ear, frame_blink_count, status = detect_blinks(frame, CONFIG['ear_threshold'])
        blink_count += frame_blink_count
        
        # Display instructions
        cv2.putText(frame, f"Blink Detection: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Time left: {CONFIG['liveness_duration'] - int(time.time() - start_time)}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Liveness Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow("Liveness Test")
    return blink_count > 0

def encode_image(image_path):
    """Encode image to base64 with proper validation"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        if not image_data:
            raise ValueError(f"Empty image file: {image_path}")
        return base64.b64encode(image_data).decode('utf-8')

def make_api_request(service, payload):
    """Make API request with proper error handling"""
    url = f"{CONFIG['server_url']}{CONFIG['endpoints'][service]}"
    headers = {
        "x-api-key": CONFIG['api_keys'][service],
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"API Error ({service}): {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except ValueError:
                print(f"Raw error response: {e.response.text}")
        return None

def capture_face(cap, image_type, perform_liveness=False):
    """Capture and process face image with optional liveness test"""
    image_path = CONFIG['image_paths'][image_type]
    
    if perform_liveness:
        if not perform_liveness_test(cap):
            print("Liveness test failed!")
            return None
        # Capture frame after successful liveness test
        ret, frame = cap.read()
        if not ret:
            return None
    else:
        # For source face, capture immediately
        ret, frame = cap.read()
        if not ret:
            return None
    
    if not cv2.imwrite(image_path, frame):
        print(f"Failed to save {image_type} image")
        return None
    
    print(f"{image_type.capitalize()} image saved to {image_path}")
    
    # Face detection and recognition (your existing code)
    print(f"Sending {image_type} for face detection...")
    detection_payload = {"file": encode_image(image_path)}
    detection_response = make_api_request("detection", detection_payload)
    
    if not detection_response or not detection_response.get("result"):
        print("Face detection failed")
        return None
    
    print("Detection successful, sending for recognition...")
    recognition_payload = {"file": encode_image(image_path)}
    recognition_response = make_api_request("recognition", recognition_payload)
    
    return recognition_response

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    source_data = None
    last_capture_time = 0
    capture_delay = 2  # seconds between captures
    
    print("Press 's' to capture source face (no liveness)")
    print("Press 't' to capture target face (with liveness test)")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            cv2.imshow("Face Capture", frame)
            
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            if key == ord('q'):
                break
            elif key == ord('s') and (current_time - last_capture_time) > capture_delay:
                last_capture_time = current_time
                source_data = capture_face(cap, "source", perform_liveness=False)
            elif key == ord('t') and (current_time - last_capture_time) > capture_delay:
                if source_data is None:
                    print("Please capture source face first (press 's')")
                    continue
                
                last_capture_time = current_time
                target_data = capture_face(cap, "target", perform_liveness=True)
                if not target_data:
                    continue
                
                print("Performing face verification...")
                verification_payload = {
                    "source_image": encode_image(CONFIG['image_paths']['source']),
                    "target_image": encode_image(CONFIG['image_paths']['target'])
                }
                verification_response = make_api_request("verification", verification_payload)
                
                if verification_response:
                    print("\nVerification Results:")
                    print(json.dumps(verification_response, indent=2))
                    
                    if verification_response.get("result"):
                        try:
                            first_result = verification_response["result"][0]
                            if first_result.get("face_matches"):
                                best_match = first_result["face_matches"][0]
                                similarity = best_match.get("similarity", 0)
                                similarity_percent = similarity * 100
                                print(f"\nFaces are {similarity_percent:.2f}% similar")
                                
                                if similarity >= CONFIG['verification_threshold']:
                                    print("✅ Verification SUCCESS: Faces match!")
                                else:
                                    print("❌ Verification FAILED: Faces don't match!")
                            else:
                                print("No face matches found in the response")
                        except (IndexError, KeyError) as e:
                            print(f"Error parsing verification response: {str(e)}")
                    else:
                        print("No verification result in the response")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
