"""
Real-Time Everyday Object Detector for Drone & Webcam (PyTorch SSDLite MobileNetV3)
Detects everyday objects (sports ball, bottle, cup, cell phone, book, laptop, chair, etc.)
and draws bounding boxes, labels, and confidence scores in real time.
"""

import cv2
import time
import torch
import numpy as np
import torchvision.models.detection as detection
from torchvision.transforms import functional as F

# COCO 80 Category Labels
COCO_CLASSES = [
 '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Highlighted everyday target objects
EVERYDAY_TARGETS = {'sports ball', 'bottle', 'cup', 'cell phone', 'book', 'laptop', 'chair', 'mouse', 'keyboard', 'scissors', 'apple', 'banana'}

class RealTimeObjectDetector:
 def __init__(self, confidence_threshold=0.45):
 self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print(f">> Initializing PyTorch SSDLite MobileNetV3 Object Detector on {self.device}...")

 weights = detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
 self.model = detection.ssdlite320_mobilenet_v3_large(weights=weights)
 self.model.to(self.device)
 self.model.eval()
 self.threshold = confidence_threshold

 def detect_frame(self, cv2_frame):
 """
 Runs object detection on a BGR OpenCV image frame.
 Returns:
 annotated_frame: frame with bounding boxes & labels drawn
 detections: list of dicts [{'label': str, 'score': float, 'box': [x1, y1, x2, y2]}]
 """
 if cv2_frame is None:
 return None, []

 height, width, _ = cv2_frame.shape
 rgb_frame = cv2_frame[:, :, ::-1]
 img_tensor = F.to_tensor(rgb_frame).unsqueeze(0).to(self.device)

 with torch.no_grad():
 predictions = self.model(img_tensor)[0]

 boxes = predictions['boxes'].cpu().numpy()
 labels = predictions['labels'].cpu().numpy()
 scores = predictions['scores'].cpu().numpy()

 annotated_frame = cv2_frame.copy()
 detections = []

 for box, label_idx, score in zip(boxes, labels, scores):
 if score >= self.threshold and label_idx < len(COCO_CLASSES):
 class_name = COCO_CLASSES[label_idx]
 if class_name == 'N/A':
 continue

 x1, y1, x2, y2 = map(int, box)
 detections.append({
 'label': class_name,
 'score': float(score),
 'box': [x1, y1, x2, y2]
 })

 # Draw bounding box & HUD tag
 is_target = class_name in EVERYDAY_TARGETS
 color = (0, 255, 0) if is_target else (255, 180, 0)
 thickness = 2 if is_target else 1

 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

 tag = f"{class_name.upper()}: {int(score * 100)}%"
 (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
 cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + tw + 6, y1), color, -1)
 cv2.putText(annotated_frame, tag, (x1 + 3, y1 - 5),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

 return annotated_frame, detections

def run_webcam_detector():
 print("=== Everyday Object Bounding-Box Detector (Webcam Mode) ===")
 detector = RealTimeObjectDetector(confidence_threshold=0.45)

 cap = cv2.VideoCapture(0)
 if not cap.isOpened():
 print("!! Could not open webcam.")
 return

 print("\n[REAL-TIME OBJECT DETECTOR ONLINE]")
 print("Point your webcam at everyday objects (ball, bottle, cup, phone, book, chair, laptop, etc.).")
 print("Press 'q' or ESC to quit.\n")

 fps_count = 0
 start_t = time.time()

 while True:
 ret, frame = cap.read()
 if not ret or frame is None:
 break

 fps_count += 1
 annotated, detections = detector.detect_frame(frame)

 # Draw HUD header
 cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (20, 20, 20), -1)
 elapsed = time.time() - start_t
 fps = fps_count / elapsed if elapsed > 0 else 0
 hud_text = f"EVERYDAY OBJECT DETECTOR (PyTorch MobileNetV3 SSDLite) | FPS: {fps:.1f} | Detections: {len(detections)}"
 cv2.putText(annotated, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

 cv2.imshow("Everyday Object Detection", annotated)
 key = cv2.waitKey(1) & 0xFF
 if key == ord('q') or key == 27:
 break

 cap.release()
 cv2.destroyAllWindows()

if __name__ == '__main__':
 run_webcam_detector()
