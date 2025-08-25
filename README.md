# Automatic-License-Plate-Recognition-ALPR-on-Raspberry-Pi
Lightweight Automatic License Plate Recognition (ALPR) system for Raspberry Pi and PC. Uses YOLOv8 (ONNX) for plate detection + EasyOCR for recognition. Supports images, video, RTSP streams, JSON/multipart webhook output, and customizable settings via .env.

alpr-pi/
├─ src/
│  ├─ alpr_v7.py               # ili alpr_v10.py / alpr_v15.py (odaberi jednu “clean” varijantu)
│  └─ utils/                   # (opciono) pomoćne funkcije
├─ models/
│  └─ lp_yolov8n.onnx          # ako ide u repo, bolje preko Git LFS (vidi dole)
├─ examples/
│  ├─ test1.png
│  └─ plates.mp4               # kratki klip (ili link u README)
├─ docs/
│  ├─ screenshot_payload.png
│  ├─ screenshot_plate.jpg
│  └─ demo.gif                 # kratak GIF umesto velikog videa
├─ .env.example                # primer konfiguracije (bez tajni)
├─ .gitignore
├─ requirements.txt
├─ README.md
└─ LICENSE                     # npr. MIT
