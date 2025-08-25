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

Download YOLOv8 ONNX model for license plates and put it in models/:

Example: lp_yolov8n.onnx

▶️ Usage

Run on a single image:
python plate_recognition.py --source test.jpg

Run on video or RTSP stream:
python plate_recognition.py --source plates.mp4 --fps 2

⚙️ Example .env
# Model & output
ONNX_MODEL=models/lp_yolov8n.onnx
OUT_DIR=out
IMG_SIZE=640

# Detection thresholds
CONF_THRESH=0.40
IOU_THRESH=0.50

# OCR
PAD_RATIO=0.18
OCR_UPSCALE=4
OCR_MIN_CONF=0.35

# Runtime
CAMERA_NAME=GateCam-1
PROC_FPS=2.0
MAX_WIDTH=1280

# De-duplication
UNIQUE_PLATES=true
UNIQUE_TTL=60

# Webhook (leave empty = FAKE)
WEBHOOK_URL=
WEBHOOK_USER=
WEBHOOK_PASS=

📦 Output
Each detection creates:
•	Cropped plate image → out/<timestamp>_plate.jpg
•	Debug frame with bbox → out/<timestamp>_frame.jpg
•	JSON payload → out/<timestamp>.json
Example payload:
{
  "time_utc": "2025-08-21T06:13:43+00:00",
  "camera_name": "GateCam-1",
  "plate_text": "NNY442",
  "confidence": 0.72,
  "detector": "YOLOv8-EasyOCR"
}

🔧 Built for lightweight real-time ALPR on edge devices.
Perfect as a starting point for IoT projects, parking systems, or smart surveillance.

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
