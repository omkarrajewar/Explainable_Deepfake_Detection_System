

# 🔍 EDDS – Explainable Deepfake Detection System

An AI-powered system designed to detect **deepfake images and videos** and provide visual explanations of the regions that may have been manipulated.

## 🚀 Overview

**EDDS (Explainable Deepfake Detection System)** combines deep learning with explainable AI techniques to make deepfake detection more understandable and transparent.

The system not only predicts whether a media file is **Real or Fake**, but also provides visual insights into the areas that influenced the prediction.

## ✨ Key Features

* 🔍 Deepfake detection for images and videos
* 🧠 Deep learning-based detection
* 🎯 **Grad-CAM** for visual explanations
* 🙂 **MediaPipe FaceMesh** for facial region analysis
* 🧪 **Error Level Analysis (ELA)** for forensic analysis
* 🌐 Flask-based web application
* 🔐 Secure user login
* 📤 Image and video upload functionality
* 📊 Explainable detection results

## 🛠️ Technologies Used

### Programming Languages

* Python

### Frameworks & Libraries

* TensorFlow
* PyTorch
* Flask
* OpenCV
* MediaPipe
* NumPy
* Pillow
* Matplotlib

### AI/ML Techniques

* Deep Learning
* XceptionNet
* Grad-CAM
* MediaPipe FaceMesh
* Error Level Analysis (ELA)

## 🔄 System Workflow

```text
User
  ↓
Login
  ↓
Upload Image / Video
  ↓
Media Processing
  ↓
Face Detection & Preprocessing
  ↓
Deep Learning Model
  ↓
Real / Fake Prediction
  ↓
Explainability Analysis
  ├── Grad-CAM
  ├── MediaPipe FaceMesh
  └── ELA
  ↓
Final Result & Visualization
```

## 📸 How It Works

1. The user logs into the application.
2. The user uploads an image or video.
3. The system preprocesses the uploaded media.
4. Facial regions are detected and analyzed.
5. The deep learning model predicts whether the media is real or manipulated.
6. Grad-CAM highlights important regions contributing to the prediction.
7. MediaPipe FaceMesh helps analyze facial regions.
8. ELA provides additional forensic information.
9. The final prediction and visual explanation are displayed to the user.

## 🎯 Objective

The main objective of EDDS is to develop a deepfake detection system that is not only accurate but also **explainable and user-friendly**.

Instead of simply displaying a **Real/Fake** result, the system provides visual evidence to help users understand why a particular prediction was made.

## 💡 Applications

* Digital media verification
* Social media content analysis
* Fake image/video detection
* Digital forensics
* Media authenticity verification
* Research in Explainable AI

## 🔮 Future Scope

* Improve detection accuracy on unseen deepfake techniques
* Support additional deepfake datasets
* Improve robustness against compressed and low-quality media
* Add real-time video detection
* Deploy the system as a scalable cloud application
* Explore advanced explainable AI techniques

## 👨‍💻 Author

**Omkar Rajewar**

B.Tech CSE – Artificial Intelligence & Machine Learning

**Interests:** Java Full Stack Development | AI/ML | Deep Learning | Software Development

## ⭐ Project

If you find this project useful, consider giving it a ⭐ on GitHub.
