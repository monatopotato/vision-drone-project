"""
Tello EDU Real-Time Room Detection & Flight Assistant
Streams camera feed from Tello EDU drone, performs real-time PyTorch room identification
(Living Room, Kitchen, Bedroom, Bathroom, Office, Dining Room), displays HUD predictions,
and logs detected room locations.

Controls:
  's' - Save snapshot & log room detection
  'f' - Launch hover flight sequence with room detection
  'q' or ESC - Exit
"""

import os
import cv2
import time
import threading
from datetime import datetime
from tello import Tello
from room_classifier import RoomClassifier

# Global state
latest_frame = None
keep_running = True
current_room = "scanning..."
current_confidence = 0.0
room_probs = {}

def draw_hud(frame, room, confidence, probs):
    """
    Draws a visual HUD overlay with room prediction, confidence gauge, and probabilities.
    """
    hud = frame.copy()
    height, width, _ = hud.shape

    # Top Banner Background
    cv2.rectangle(hud, (0, 0), (width, 80), (20, 20, 20), -1)
    
    # Title & Prediction Text
    cv2.putText(hud, "TELLO AI ROOM DETECTOR", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    color = (0, 255, 0) if confidence > 0.6 else (0, 215, 255)
    pred_str = f"ROOM: {room.upper()} ({confidence*100:.1f}%)"
    cv2.putText(hud, pred_str, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Right side mini probability bars
    bar_x = width - 210
    bar_y = 20
    cv2.rectangle(hud, (bar_x - 10, 10), (width - 10, 160), (30, 30, 30), -1)
    
    for idx, (cat, prob) in enumerate(probs.items()):
        y_pos = bar_y + (idx * 20)
        label = f"{cat[:7]}: {int(prob*100)}%"
        cv2.putText(hud, label, (bar_x, y_pos + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Bar chart line
        bar_w = int(prob * 70)
        cv2.rectangle(hud, (bar_x + 95, y_pos + 4), (bar_x + 95 + bar_w, y_pos + 12), (0, 255, 120), -1)

    # Controls footer
    cv2.putText(hud, "[S] Log Room Snap  |  [F] Launch Hover  |  [Q] Quit", (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return hud

def inference_worker(classifier):
    """
    Background worker thread running PyTorch model inference on camera frames.
    """
    global latest_frame, keep_running, current_room, current_confidence, room_probs
    while keep_running:
        if latest_frame is not None:
            frame_copy = latest_frame.copy()
            room, conf, probs = classifier.predict_frame(frame_copy)
            current_room = room
            current_confidence = conf
            room_probs = probs
        time.sleep(0.1)  # Run inference ~10 FPS

def video_worker(video_url):
    """
    Background worker thread pulling H.264 camera frames from Tello UDP stream.
    """
    global latest_frame, keep_running
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    while keep_running:
        ret, frame = cap.read()
        if ret and frame is not None:
            latest_frame = frame
        else:
            time.sleep(0.01)
    cap.release()

def main():
    global latest_frame, keep_running, current_room, current_confidence, room_probs
    
    os.makedirs('room_detections', exist_ok=True)
    print("=== Tello EDU AI Room Recognition System ===")

    # 1. Initialize PyTorch Room Classifier
    custom_model_path = os.path.join("models", "room_model.pth")
    classifier = RoomClassifier(model_path=custom_model_path if os.path.exists(custom_model_path) else None)

    # 2. Connect to Tello Drone
    drone = Tello()
    if not drone.send_command('command'):
        print("!! Could not connect to Tello. Verify Wi-Fi network TELLO-XXXXXX.")
        drone.close()
        return

    drone.stream_on()
    time.sleep(1.5)

    video_url = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
    
    # 3. Start Threads
    v_thread = threading.Thread(target=video_worker, args=(video_url,))
    v_thread.daemon = True
    v_thread.start()

    i_thread = threading.Thread(target=inference_worker, args=(classifier,))
    i_thread.daemon = True
    i_thread.start()

    print("\n[AI ROOM DETECTOR ONLINE]")
    print("Controls: 's' = Save Detection Snap | 'f' = Launch Flight | 'q' = Quit\n")

    try:
        while keep_running:
            if latest_frame is not None:
                hud = draw_hud(latest_frame, current_room, current_confidence, room_probs)
                cv2.imshow("Tello AI Room Detection", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            elif key == ord('s'):
                if latest_frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_name = f"detected_{current_room}_{timestamp}.jpg"
                    save_path = os.path.join('room_detections', snap_name)
                    cv2.imwrite(save_path, latest_frame)
                    print(f"[SNAPSHOT] Logged Room Detection [{current_room.upper()} ({current_confidence*100:.1f}%)]: {save_path}")

            elif key == ord('f'):
                print("\n[FLIGHT] Autonomous Room Survey Flight Initiated!")
                drone.send_command('takeoff')
                time.sleep(3)
                drone.send_command('cw 90')
                time.sleep(2)
                drone.send_command('ccw 90')
                time.sleep(2)
                drone.send_command('land')
                print("Flight routine finished.")

    except KeyboardInterrupt:
        print("\nEmergency Exit.")

    finally:
        keep_running = False
        cv2.destroyAllWindows()
        drone.stream_off()
        drone.close()
        print("=== Room Detector System Shutdown ===")

if __name__ == '__main__':
    main()
