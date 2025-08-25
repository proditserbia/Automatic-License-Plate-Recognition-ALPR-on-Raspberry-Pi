# Automatic-License-Plate-Recognition-ALPR-on-Raspberry-Pi
Lightweight Automatic License Plate Recognition (ALPR) system for Raspberry Pi and PC. Uses YOLOv8 (ONNX) for plate detection + EasyOCR for recognition. Supports images, video, RTSP streams, JSON/multipart webhook output, and customizable settings via .env.

# ALPR on Raspberry Pi (and PC)
Lightweight open-source ALPR pipeline based on **YOLOv8 (ONNX)** for plate detection + **EasyOCR** for text recognition.  
Supports single image, video/RTSP, and webhook delivery (JSON + images).

## Features
- YOLOv8 (ONNXRuntime) plate detection (CPU/GPU fallback)
- EasyOCR recognition with light preprocessing
- Config via `.env` (no code changes)
- Optional: de-duplication, temporal voting
- Webhook: JSON only or multipart (frame/plate/detect images)

## Quick Start
```bash
git clone https://github.com/<your-user>/alpr-pi.git
cd alpr-pi
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # set your values
