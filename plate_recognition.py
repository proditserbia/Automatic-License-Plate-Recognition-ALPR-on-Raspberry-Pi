import os, cv2, datetime, json, argparse, time, base64
import numpy as np
import onnxruntime as ort
import easyocr
import requests
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()

ONNX_MODEL     = os.getenv("ONNX_MODEL", "models/lp_yolov8n.onnx")
OUT_DIR        = "out"
CONF_THRESH    = float(os.getenv("CONF_THRESH", 0.40))
IOU_THRESH     = float(os.getenv("IOU_THRESH", 0.50))
IMG_SIZE       = int(os.getenv("IMG_SIZE", 640))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.5))
UNIQUE_PLATES  = os.getenv("UNIQUE_PLATES", "true").lower() == "true"

os.makedirs(OUT_DIR, exist_ok=True)

# Webhook
WEBHOOK_URL  = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_USER = os.getenv("WEBHOOK_USER", "").strip()
WEBHOOK_PASS = os.getenv("WEBHOOK_PASS", "").strip()
CAMERA_NAME  = os.getenv("CAMERA_NAME", "GateCam-1")

# ---------------- ONNX Runtime ----------------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
print(f"[INFO] ONNX providers preference: {providers}")
session = ort.InferenceSession(ONNX_MODEL, providers=providers)
inp_name  = session.get_inputs()[0].name
out_name  = session.get_outputs()[0].name

# ---------------- OCR ----------------
ocr = easyocr.Reader(['en'], gpu=('CUDAExecutionProvider' in session.get_providers()))

# ---------------- Utils ----------------
def ts_iso():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

def ts_safe():
    return ts_iso().replace(":", "-")

def letterbox(im, new_shape=(IMG_SIZE, IMG_SIZE), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    nw, nh = int(round(w*r)), int(round(h*r))
    im_res = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1]-nw, new_shape[0]-nh
    left, right  = int(round(dw/2-0.1)), int(round(dw/2+0.1))
    top, bottom  = int(round(dh/2-0.1)), int(round(dh/2+0.1))
    out = cv2.copyMakeBorder(im_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def nms(boxes, scores, iou_thr=0.5):
    if not boxes: return []
    boxes = np.array(boxes); scores = np.array(scores)
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep=[]
    while order.size>0:
        i = order[0]; keep.append(i)
        if order.size==1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1); h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        ovr = inter/(areas[i]+areas[order[1:]]-inter)
        order = order[1:][ovr<=iou_thr]
    return keep

def run_webhook(payload: dict):
    if not WEBHOOK_URL:
        print("[Webhook skipped]")
        return
    headers = {"Content-Type": "application/json"}
    if WEBHOOK_USER or WEBHOOK_PASS:
        token = base64.b64encode(f"{WEBHOOK_USER}:{WEBHOOK_PASS}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    try:
        r = requests.post(WEBHOOK_URL, headers=headers, json=payload, timeout=5)
        print(f"[Webhook] {r.status_code}")
    except Exception as e:
        print(f"[Webhook error] {e}")

# ---------------- Detection ----------------
def detect_plate(frame_bgr):
    H, W = frame_bgr.shape[:2]
    img, r, (dw, dh) = letterbox(frame_bgr, (IMG_SIZE, IMG_SIZE))
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))[None, ...]

    raw = session.run([out_name], {inp_name: x})[0]
    preds = np.squeeze(raw).transpose()

    boxes, scores = [], []
    for det in preds:
        conf = det[4]
        if conf < CONF_THRESH: 
            continue
        cx,cy,w,h = det[:4]
        x1_ = cx - w/2; y1_ = cy - h/2; x2_ = cx + w/2; y2_ = cy + h/2
        x1 = (x1_ - dw) / r; y1 = (y1_ - dh) / r
        x2 = (x2_ - dw) / r; y2 = (y2_ - dh) / r
        x1 = int(max(0, min(W-1, x1))); y1 = int(max(0, min(H-1, y1)))
        x2 = int(max(0, min(W-1, x2))); y2 = int(max(0, min(H-1, y2)))
        if x2> x1 and y2> y1:
            boxes.append([x1,y1,x2,y2]); scores.append(float(conf))

    if not boxes: return frame_bgr, None, 0.0
    keep = nms(boxes, scores, IOU_THRESH)
    best_idx = int(np.argmax([scores[i] for i in keep]))
    bi = keep[best_idx]
    x1,y1,x2,y2 = boxes[bi]; conf = scores[bi]
    crop = frame_bgr[y1:y2, x1:x2]
    dbg  = frame_bgr.copy()
    cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
    return dbg, crop, conf

def ocr_text(plate_bgr):
    if plate_bgr is None or plate_bgr.size == 0: return ""
    res = ocr.readtext(plate_bgr, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")
    if not res: return ""
    best = max(res, key=lambda r: r[2])
    if best[2] < MIN_CONFIDENCE: return ""
    return best[1].strip()

# ---------------- Main loop ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path | RTSP URL | camera index (e.g. 0)")
    ap.add_argument("--fps", type=float, default=2.0, help="processing FPS (default 2)")
    ap.add_argument("--max-width", type=int, default=1280, help="resize width (0=off)")
    args = ap.parse_args()

    src = args.source
    if src.isdigit(): src = int(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    frame_interval = 1.0 / max(0.1, args.fps)
    last_t = 0.0
    seen_plates = set()

    print(f"[INFO] Processing at ~{args.fps} FPS. Webhook: {'ON' if WEBHOOK_URL else 'OFF'}")

    while True:
        ok, frame = cap.read()
        if not ok: break

        if args.max_width and frame.shape[1] > args.max_width:
            scale = args.max_width / frame.shape[1]
            frame = cv2.resize(frame, (args.max_width, int(frame.shape[0]*scale)))

        now = time.time()
        if now - last_t < frame_interval: continue
        last_t = now

        debug_img, crop, det_conf = detect_plate(frame)
        text = ocr_text(crop) if crop is not None else ""
        if not text: continue

        if UNIQUE_PLATES and text in seen_plates: continue
        seen_plates.add(text)

        ts, tsf = ts_iso(), ts_safe()

        if debug_img is not None:
            cv2.imwrite(os.path.join(OUT_DIR, f"{tsf}_frame.jpg"), debug_img)
        if crop is not None:
            cv2.imwrite(os.path.join(OUT_DIR, f"{tsf}_plate.jpg"), crop)

        payload = {
            "time_utc": ts,
            "camera_name": CAMERA_NAME,
            "plate_text": text,
            "confidence": float(det_conf),
            "detector": "YOLOv8-ONNX+EasyOCR"
        }

        with open(os.path.join(OUT_DIR, f"{tsf}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        run_webhook(payload)

    cap.release()

if __name__ == "__main__":
    main()
