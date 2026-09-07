"""
PC Webcam AI Room Detector Test Script
Tests the PyTorch room classification model using your computer's built-in webcam (or USB webcam)
without requiring the Tello drone to be connected.

Controls:
  's' - Save snapshot & log room detection to room_detections/
  'q' or ESC - Exit viewer
"""

import os
import cv2
import time
import torch
from datetime import datetime
from room_classifier import RoomClassifier

def draw_hud(frame, room, confidence, probs):
    """
    Draws a visual HUD overlay with room prediction, confidence gauge, and probabilities.
    """
    hud = frame.copy()
    height, width, _ = hud.shape

    # Top Banner Background
    cv2.rectangle(hud, (0, 0), (width, 80), (20, 20, 20), -1)
    
    # Title & Prediction Text
    cv2.putText(hud, "PC WEBCAM - AI ROOM DETECTOR", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    color = (0, 255, 0) if confidence > 0.5 else (0, 215, 255)
    pred_str = f"ROOM: {room.upper()} ({confidence*100:.1f}%)"
    cv2.putText(hud, pred_str, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Right side mini probability bars
    bar_x = width - 210
    bar_y = 15
    cv2.rectangle(hud, (bar_x - 10, 5), (width - 10, 160), (30, 30, 30), -1)
    
    for idx, (cat, prob) in enumerate(probs.items()):
        y_pos = bar_y + (idx * 22)
        label = f"{cat[:7]}: {int(prob*100)}%"
        cv2.putText(hud, label, (bar_x, y_pos + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Bar chart line
        bar_w = int(prob * 70)
        cv2.rectangle(hud, (bar_x + 95, y_pos + 4), (bar_x + 95 + bar_w, y_pos + 12), (0, 255, 120), -1)

    # Controls footer
    cv2.putText(hud, "[S] Save Snapshot  |  [Q] Quit", (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return hud

def main():
    os.makedirs('room_detections', exist_ok=True)
    print("=== PC Webcam AI Room Detector Test ===")

    # 1. Initialize PyTorch Room Classifier
    custom_model_path = os.path.join("models", "room_model.pth")
    if os.path.exists(custom_model_path):
        print(f"[INFO] Using custom trained model: '{custom_model_path}'")
        classifier = RoomClassifier(model_path=custom_model_path)
    else:
        print("[INFO] Custom model not found, using pre-trained ResNet18 model.")
        classifier = RoomClassifier()

    # 2. Open PC Webcam (Device Index 0)
    print("[INFO] Accessing computer webcam (Index 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("![ERROR] Could not open computer webcam. Verify camera permissions and connection.")
        return

    print("\n[WEBCAM DETECTOR ONLINE]")
    print("Point your webcam around your room to see real-time predictions!")
    print("Controls: 's' = Save Snapshot | 'q' = Quit\n")

    current_room = "scanning..."
    current_confidence = 0.0
    room_probs = {}

    last_infer_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("![WARNING] Failed to grab frame from webcam.")
                time.sleep(0.1)
                continue

            # Run inference ~5-10 times per second for smooth video feed
            now = time.time()
            if (now - last_infer_time) > 0.1:
                current_room, current_confidence, room_probs = classifier.predict_frame(frame)
                last_infer_time = now

            # Draw visual HUD overlay
            hud = draw_hud(frame, current_room, current_confidence, room_probs)
            cv2.imshow("PC Webcam - AI Room Detector", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_name = f"webcam_{current_room}_{timestamp}.jpg"
                save_path = os.path.join('room_detections', snap_name)
                cv2.imwrite(save_path, frame)
                print(f"[SNAPSHOT] Logged Room Detection [{current_room.upper()} ({current_confidence*100:.1f}%)]: {save_path}")

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("=== PC Webcam Detector Closed ===")

if __name__ == '__main__':
    main()
