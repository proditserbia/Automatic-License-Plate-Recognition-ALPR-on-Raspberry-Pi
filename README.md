# Automatic License Plate Recognition (ALPR)

Lightweight open-source ALPR system that works on **Raspberry Pi** and standard PCs.  
Uses **YOLOv8 (ONNX)** for license plate detection and **EasyOCR** for character recognition.

## ✨ Features
- 📸 Supports images, video files, RTSP streams, and webcams  
- ⚡ Lightweight, runs on CPU (GPU optional)  
- 🔧 Configurable via `.env` file  
- 📤 Webhook support (JSON or multipart with plate & frame images)  
- 🛡️ Unique plate filtering (avoid duplicates within TTL)  
- 💾 Local saving of frames, crops, and JSON payloads in `out/`

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/prodit/plate_recognition.git
cd plate_recognition

# Install dependencies
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
