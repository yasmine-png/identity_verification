# Tunisian National ID Card Processing and Verification System

## Project Overview

This project implements an advanced automated system for processing Tunisian National ID Cards by leveraging state-of-the-art computer vision and deep learning techniques. The solution focuses on identity verification through facial recognition and automatic data extraction from ID cards, aiming to enhance security, accuracy, and user experience in identity validation workflows.

---

## Problem Statement

Reliable, rapid, and secure identity verification remains a critical challenge for many organizations. Manual checks are often error-prone and susceptible to fraud. This system addresses these challenges by automating identity verification processes using Artificial Intelligence (AI), thereby reducing human error and minimizing fraudulent activities.

---

## Target Users and Use Cases

### Target Users

- **Administrators:** Oversee and audit identity verification processes  
- **End Users:** Individuals submitting identity verification requests  

### Use Cases

- Secure submission of ID card images and live selfies  
- Automated detection and extraction of key identity fields  
- Robust facial comparison to detect identity fraud  
- Real-time liveness detection to prevent spoofing attacks  
- Database validation to avoid duplicate or fraudulent entries  

---

## System Architecture

### 1. Automated Data Extraction Pipeline

- **ID Card Upload:** Users upload images of their Tunisian National ID Cards.  
- **Zone Detection:** A YOLOv5-based object detection model identifies critical regions such as Name, Surname, and ID Number on the card.  
- **Region Cropping:** Detected areas are cropped for dedicated processing.  
- **Text Recognition:** PaddleOCR, optimized for Arabic and Latin scripts, extracts textual data from cropped regions.  
- **Data Storage:** Extracted information is securely stored in a MongoDB database.  
- **User Verification:** The system checks for existing entries to prevent duplicates.  

### Data Preparation & Model Training

- **Annotation:** Images are annotated using MakeSense.ai, labeling bounding boxes for name, surname, ID number, and photo regions.  
- **Training:** YOLOv5 is trained on the annotated dataset to accurately detect fields on diverse ID card images.  
- **Inference:** Upon deployment, YOLOv5 automatically detects zones on new card images, enabling automated cropping and OCR extraction.  

---

### 2. Facial Recognition and Liveness Detection Pipeline

- **Selfie Capture:** Users capture a real-time selfie via webcam.  
- **Face & Eye Detection:** OpenCV's Haar cascades detect faces and eyes within the selfie.  
- **Liveness Check:** Using dlib’s 68 facial landmark predictor, the system calculates the Eye Aspect Ratio (EAR) to detect natural blinking, confirming liveness and preventing spoofing.  
- **Image Encoding:** The selfie is encoded in Base64 for secure JSON transmission.  
- **Face Matching:** The backend queries the CompreFace API to compare the selfie against the extracted ID photo.  
- **Verification Decision:** A similarity score above 0.88 confirms identity verification. Verified images and results are saved for audit purposes.  

---

### 3. Big Data Storage with Hadoop HDFS

To handle large volumes of identity document images and scale the system for enterprise use, the project integrates **Hadoop Distributed File System (HDFS)**:

- **Distributed Storage:** Hadoop stores raw ID card images and live selfie data across a scalable cluster.  
- **Fault Tolerance:** Ensures data durability with replication and recovery mechanisms.  
- **Scalability:** Supports growth in data without loss of performance.  
- **Efficient Data Processing:** Enables batch analytics or further data operations on large datasets.

The Django backend connects seamlessly with Hadoop HDFS to store and retrieve identity documents securely, providing a robust foundation for big data management.

---

## Technical Highlights

- **YOLOv5:** State-of-the-art object detection for precise localization of ID card fields.  
- **PaddleOCR:** Deep learning-based OCR supporting Arabic and Latin scripts, optimized for distorted and curved texts typical of ID cards.  
- **OpenCV & dlib:** Real-time facial detection and landmark extraction enabling robust liveness detection through blink analysis.  
- **CompreFace API:** Open-source facial recognition API providing high-accuracy similarity scoring.  
- **MongoDB:** NoSQL database for flexible and scalable storage of extracted data and verification logs.  
- **Hadoop HDFS:** Distributed file system for secure, scalable storage of large volumes of identity documents.  

---

## Impact

This end-to-end system strengthens security for online banking, government services, and digital identity platforms by preventing fraud and enabling seamless identity verification. The project combines AI-powered object detection, OCR, facial recognition, and big data technologies to deliver a secure, efficient, and user-friendly solution.
