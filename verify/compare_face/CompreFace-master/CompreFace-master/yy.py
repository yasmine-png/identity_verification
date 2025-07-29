import cv2
import base64
import requests
import os
import json
import time
import numpy as np
from requests.exceptions import RequestException

# Configuration (unchanged)
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
        "source": "source_face.jpg",
        "target": "target_face.jpg"
    },
    "verification_threshold": 0.88,
    "liveness": {
        "eye_close_threshold": 0.10,
        "min_blinks": 3,
        "check_seconds": 10,
        "eye_cascade": cv2.data.haarcascades + 'haarcascade_eye.xml',
        "face_cascade": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    }
}

# New face visualization function only
def draw_pro_face_box(frame, x, y, w, h, status="active"):
    """Professional face box drawing only - everything else remains identical"""
    color = (0, 255, 0) if status == "active" else (0, 0, 255)
    
    # Main rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw corner markers
    corner_length = min(w, h) // 4
    # Top-left
    cv2.line(frame, (x, y), (x + corner_length, y), color, 2)
    cv2.line(frame, (x, y), (x, y + corner_length), color, 2)
    # Top-right
    cv2.line(frame, (x+w, y), (x+w - corner_length, y), color, 2)
    cv2.line(frame, (x+w, y), (x+w, y + corner_length), color, 2)
    # Bottom-left
    cv2.line(frame, (x, y+h), (x + corner_length, y+h), color, 2)
    cv2.line(frame, (x, y+h), (x, y+h - corner_length), color, 2)
    # Bottom-right
    cv2.line(frame, (x+w, y+h), (x+w - corner_length, y+h), color, 2)
    cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_length), color, 2)
    
    # Status text
    cv2.putText(frame, "FACE DETECTED", (x, y-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# EVERYTHING BELOW THIS POINT REMAINS EXACTLY THE SAME AS YOUR ORIGINAL CODE
# Only replaced cv2.rectangle calls with draw_pro_face_box where faces are drawn

def encode_image(image_path):
    """Convert image to base64 with validation"""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            if not image_data:
                print("Error: Empty image file")
                return None
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Image encoding error: {str(e)}")
        return None

def make_api_request(service, payload):
    """Make API request with timeout and error handling"""
    try:
        url = f"{CONFIG['server_url']}{CONFIG['endpoints'][service]}"
        headers = {
            "x-api-key": CONFIG['api_keys'][service],
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"API Error ({service}): {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except:
                print(f"Raw response: {e.response.text}")
        return None

def check_eyes_closed(frame, eye_cascade, face_roi):
    """Simplified blink detection using eye region analysis"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[face_roi[1]:face_roi[1]+face_roi[3], face_roi[0]:face_roi[0]+face_roi[2]]
    
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(30, 30))
    
    if len(eyes) == 0:
        # No eyes detected - likely closed
        return True
    
    # If eyes are detected but very small, consider them closed
    eye_status = []
    for (ex, ey, ew, eh) in eyes:
        eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
        _, threshold = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)
        contour_area = cv2.countNonZero(threshold)
        eye_status.append(contour_area / (ew * eh) > 0.4)  # Threshold for "closed"
    
    return all(eye_status)

def perform_liveness_check(cap):
    """More reliable liveness check with visual feedback"""
    face_cascade = cv2.CascadeClassifier(CONFIG["liveness"]["face_cascade"])
    eye_cascade = cv2.CascadeClassifier(CONFIG["liveness"]["eye_cascade"])
    
    blink_count = 0
    start_time = time.time()
    last_state = None
    
    print("\nLiveness Check Instructions:")
    print("1. Look straight at the camera")
    print("2. Blink naturally")
    print(f"3. Complete {CONFIG['liveness']['min_blinks']} blink(s) in {CONFIG['liveness']['check_seconds']} seconds")
    
    while (time.time() - start_time) < CONFIG["liveness"]["check_seconds"]:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_state = "searching"
        
        for (x, y, w, h) in faces:
            draw_pro_face_box(frame, x, y, w, h)  # Changed to new drawing function
            
            eyes_closed = check_eyes_closed(frame, eye_cascade, (x, y, w, h))
            current_state = "closed" if eyes_closed else "open"
            
            # Detect state change (open->closed)
            if last_state == "open" and current_state == "closed":
                blink_count += 1
                print(f"Blink detected! ({blink_count}/{CONFIG['liveness']['min_blinks']})")
            
            last_state = current_state
            
            # Draw eye status
            status_text = f"Eyes: {current_state.upper()}"
            cv2.putText(frame, status_text, (x, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, f"Blinks: {blink_count}/{CONFIG['liveness']['min_blinks']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time left: {int(CONFIG['liveness']['check_seconds'] - (time.time() - start_time))}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Liveness Check - Blink Naturally", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        if blink_count >= CONFIG["liveness"]["min_blinks"]:
            cv2.destroyWindow("Liveness Check - Blink Naturally")
            return True
    
    cv2.destroyWindow("Liveness Check - Blink Naturally")
    return False

def capture_and_process_face(cap, image_type, require_liveness=False):
    """Complete face capture and processing pipeline"""
    try:
        # Capture face
        if require_liveness:
            if not perform_liveness_check(cap):
                print("Liveness verification failed!")
                return None
            ret, frame = cap.read()
        else:
            ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            return None
        
        # Save image
        image_path = CONFIG['image_paths'][image_type]
        if not cv2.imwrite(image_path, frame):
            print(f"Error: Failed to save {image_type} image")
            return None
        
        print(f"\nProcessing {image_type} face...")
        
        # Face detection
        print("Performing face detection...")
        encoded = encode_image(image_path)
        if not encoded:
            return None
            
        detection = make_api_request("detection", {"file": encoded})
        if not detection or not detection.get("result"):
            print("Face detection failed - ensure face is clearly visible")
            return None
        
        # Face recognition
        print("Performing face recognition...")
        recognition = make_api_request("recognition", {"file": encoded})
        if not recognition:
            print("Face recognition failed")
            return None
            
        return recognition
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None

def main():
    print("\nFace Verification System")
    print("-----------------------")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera")
        return
    
    # Check if face detection works
    test_frame = cv2.imread(CONFIG['image_paths']['source'], cv2.IMREAD_COLOR)
    if test_frame is not None:
        face_cascade = cv2.CascadeClassifier(CONFIG["liveness"]["face_cascade"])
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        if len(face_cascade.detectMultiScale(gray, 1.3, 5)) == 0:
            print("Warning: Face detection test failed - check lighting/angle")
    
    source_data = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Camera feed interrupted")
                break
            
            # Display instructions
            cv2.putText(frame, "Press 's': Capture Source Face", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 't': Capture Target Face (Liveness)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q': Quit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show real-time face detection with new style
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(CONFIG["liveness"]["face_cascade"])
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            for (x, y, w, h) in faces:
                draw_pro_face_box(frame, x, y, w, h)
            
            cv2.imshow("Face Verification System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                source_data = capture_and_process_face(cap, "source")
                if source_data:
                    print("Source face captured and processed successfully!")
            elif key == ord('t'):
                if not source_data:
                    print("Error: Capture source face first!")
                    continue
                
                target_data = capture_and_process_face(cap, "target", require_liveness=True)
                if not target_data:
                    continue
                
                print("\nVerifying faces...")
                verification = make_api_request("verification", {
                    "source_image": encode_image(CONFIG['image_paths']['source']),
                    "target_image": encode_image(CONFIG['image_paths']['target'])
                })
                
                if verification and verification.get("result"):
                    similarity = verification["result"][0]["face_matches"][0]["similarity"]
                    print(f"\nVerification Result: {similarity*100:.2f}% similarity")
                    if similarity >= CONFIG["verification_threshold"]:
                        print("✅ VERIFICATION SUCCESS")
                    else:
                        print("❌ VERIFICATION FAILED")
                else:
                    print("Verification failed - check API connection")
    
    except Exception as e:
        print(f"\nSystem error: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSystem shutdown")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
