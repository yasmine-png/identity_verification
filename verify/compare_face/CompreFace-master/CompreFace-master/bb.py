import cv2
import base64
import requests
import os
import json
import time
import numpy as np
from requests.exceptions import RequestException

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
    "verification_threshold": 0.7,
    "liveness": {
        "min_eye_frames": 3,     # Minimum frames with eyes detected
        "check_seconds": 5,      # Duration of liveness check
        "eye_cascade": cv2.data.haarcascades + 'haarcascade_eye.xml'
    }
}

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

def check_liveness(cap):
    """Perform liveness check by detecting eyes"""
    eye_cascade = cv2.CascadeClassifier(CONFIG['liveness']['eye_cascade'])
    start_time = time.time()
    eye_frames = 0
    
    print("Liveness check: Please look straight at the camera")
    
    while (time.time() - start_time) < CONFIG['liveness']['check_seconds']:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Draw eye rectangles and count frames with eyes detected
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        if len(eyes) >= 2:  # At least two eyes detected
            eye_frames += 1
        
        # Display instructions
        cv2.putText(frame, f"Liveness Check: {int(CONFIG['liveness']['check_seconds'] - (time.time() - start_time))}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Liveness Check", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow("Liveness Check")
    return eye_frames >= CONFIG['liveness']['min_eye_frames']

def capture_face(cap, image_type, perform_liveness=False):
    """Capture and process face image with optional liveness check"""
    if perform_liveness:
        if not check_liveness(cap):
            print("Liveness verification failed!")
            return None
        ret, frame = cap.read()  # Capture frame after liveness check
        if not ret:
            return None
    else:
        ret, frame = cap.read()  # Normal capture for source face
        if not ret:
            return None
    
    image_path = CONFIG['image_paths'][image_type]
    if not cv2.imwrite(image_path, frame):
        print(f"Failed to save {image_type} image")
        return None
    
    print(f"{image_type.capitalize()} image saved to {image_path}")
    
    # Original detection and recognition logic
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

def draw_face_rectangle(frame, face_data):
    """Draw rectangle around detected face"""
    if not face_data or not face_data.get("result"):
        return frame
    
    try:
        face_box = face_data["result"][0]["box"]
        x_min, y_min = face_box["x_min"], face_box["y_min"]
        x_max, y_max = face_box["x_max"], face_box["y_max"]
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    except (KeyError, IndexError):
        pass
    
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    source_data = None
    last_capture_time = 0
    capture_delay = 2  # seconds between captures
    
    print("Press 's' to capture source face (no liveness)")
    print("Press 't' to capture target face (with liveness check)")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Display frame with face rectangle if available
            display_frame = frame.copy()
            if source_data:
                display_frame = draw_face_rectangle(display_frame, source_data)
            
            cv2.imshow("Face Capture", display_frame)
            
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
