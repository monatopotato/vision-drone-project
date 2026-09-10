"""
Tello EDU Real-Time Everyday Object Detector (PyTorch MobileNetV3 SSDLite)
Streams live video feed from Tello drone, detects everyday objects (ball, bottle, cup, phone, book, laptop, chair, etc.),
draws bounding boxes and labels on HUD, and logs detections.
"""

import os
import cv2
import time
import threading
from datetime import datetime

try:
    from src.drone.tello import Tello
    from src.detection.real_time_object_detector import RealTimeObjectDetector
except ImportError:
    from tello import Tello
    from real_time_object_detector import RealTimeObjectDetector

latest_frame = None
annotated_frame = None
keep_running = True
current_detections = []

def inference_worker(detector):
    global latest_frame, annotated_frame, keep_running, current_detections
    while keep_running:
        if latest_frame is not None:
            frame_copy = latest_frame.copy()
            ann, detections = detector.detect_frame(frame_copy)
            annotated_frame = ann
            current_detections = detections
        time.sleep(0.05)

def video_worker(video_url):
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
    global latest_frame, annotated_frame, keep_running, current_detections
    
    os.makedirs('object_detections', exist_ok=True)
    print("=== Tello EDU Everyday Object Detector (PyTorch MobileNetV3 SSDLite) ===")

    detector = RealTimeObjectDetector(confidence_threshold=0.45)
    drone = Tello()
    
    if not drone.send_command('command'):
        print("!! Could not connect to Tello. Verify Wi-Fi network TELLO-XXXXXX.")
        drone.close()
        return

    drone.stream_on()
    time.sleep(1.5)
    video_url = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
    
    v_thread = threading.Thread(target=video_worker, args=(video_url,))
    v_thread.daemon = True
    v_thread.start()

    i_thread = threading.Thread(target=inference_worker, args=(detector,))
    i_thread.daemon = True
    i_thread.start()

    print("\n[AI EVERYDAY OBJECT DETECTOR ONLINE]")
    print("Controls: 's' = Save Snapshot | 'f' = Launch Flight | 'q' = Quit\n")

    try:
        while keep_running:
            display_frame = annotated_frame if annotated_frame is not None else latest_frame
            if display_frame is not None:
                hud = display_frame.copy()
                cv2.rectangle(hud, (0, 0), (hud.shape[1], 45), (20, 20, 20), -1)
                det_summary = ", ".join([f"{d['label'].upper()} ({int(d['score']*100)}%)" for d in current_detections[:3]])
                status_txt = f"TELLO OBJECT DETECTOR | Detections: {len(current_detections)} | {det_summary}"
                cv2.putText(hud, status_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)
                
                cv2.imshow("Tello AI Everyday Object Detector", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            elif key == ord('s'):
                if display_frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_name = f"object_detect_{timestamp}.jpg"
                    save_path = os.path.join('object_detections', snap_name)
                    cv2.imwrite(save_path, display_frame)
                    print(f"[SNAPSHOT] Saved object detection image: {save_path}")

            elif key == ord('f'):
                print("\n[FLIGHT] Autonomous Survey Flight Started!")
                drone.send_command('takeoff')
                time.sleep(3)
                drone.send_command('cw 90')
                time.sleep(2)
                drone.send_command('ccw 90')
                time.sleep(2)
                drone.send_command('land')
                print("Flight routine complete.")

    except KeyboardInterrupt:
        print("\nEmergency Exit.")

    finally:
        keep_running = False
        cv2.destroyAllWindows()
        drone.stream_off()
        drone.close()
        print("=== Detector Shutdown ===")

if __name__ == '__main__':
    main()
