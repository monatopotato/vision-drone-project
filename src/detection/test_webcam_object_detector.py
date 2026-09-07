"""
Webcam Everyday Object Detector
Tests PyTorch MobileNetV3 Everyday Object Classifier (ball, bottle, cup, phone, book, chair, laptop)
using your computer's webcam feed.
"""

import cv2
import os
import time

try:
 from src.detection.object_classifier import ObjectClassifier
except ImportError:
 from object_classifier import ObjectClassifier

def main():
 print("=== Everyday Object Detector (Webcam Live Mode) ===")

 model_path = os.path.join("models", "object_model.pth")
 classifier = ObjectClassifier(model_path=model_path if os.path.exists(model_path) else None)

 cap = cv2.VideoCapture(0)
 if not cap.isOpened():
 print("!! Could not open webcam (index 0). Make sure webcam is connected.")
 return

 print("\n[OBJECT DETECTOR ONLINE]")
 print("Point your webcam at objects like a Ball, Bottle, Cup, Phone, Book, Laptop, or Chair.")
 print("Press 'q' or ESC to quit.\n")

 while True:
 ret, frame = cap.read()
 if not ret or frame is None:
 print("!! Failed to grab webcam frame.")
 break

 top_obj, confidence, probs = classifier.predict_frame(frame)

 hud = frame.copy()
 height, width, _ = hud.shape

 cv2.rectangle(hud, (0, 0), (width, 70), (20, 20, 20), -1)
 color = (0, 255, 0) if confidence > 0.6 else (0, 215, 255)
 cv2.putText(hud, "EVERYDAY OBJECT DETECTOR", (15, 25),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
 cv2.putText(hud, f"OBJECT: {top_obj.upper()} ({confidence*100:.1f}%)", (15, 55),
 cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

 bar_x = width - 200
 cv2.rectangle(hud, (bar_x - 10, 10), (width - 10, 170), (30, 30, 30), -1)
 for idx, (cat, prob) in enumerate(probs.items()):
 y_pos = 25 + (idx * 20)
 label = f"{cat[:7]}: {int(prob*100)}%"
 cv2.putText(hud, label, (bar_x, y_pos),
 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
 bar_w = int(prob * 60)
 cv2.rectangle(hud, (bar_x + 85, y_pos - 8), (bar_x + 85 + bar_w, y_pos), (0, 255, 120), -1)

 cv2.imshow("Everyday Object Recognition", hud)

 key = cv2.waitKey(1) & 0xFF
 if key == ord('q') or key == 27:
 break

 cap.release()
 cv2.destroyAllWindows()
 print("=== Object Detector Closed ===")

if __name__ == '__main__':
 main()
