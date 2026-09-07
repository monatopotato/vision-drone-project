"""
Tello EDU Flight with Real-Time Camera Stream & Automated Waypoint Snapshots

Executes the flight sequence (takeoff, hover, left, right, up, down, land)
while streaming live video to screen and automatically saving snapshot photos
at each way-point!
"""

import os
import cv2
import time
import threading
from datetime import datetime
from tello import Tello

# Shared state for thread frame reading
latest_frame = None
keep_running = True

def video_stream_worker(video_url):
 """
 Background worker thread to continuously pull H.264 frames from Tello UDP stream.
 """
 global latest_frame, keep_running
 cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

 while keep_running:
 ret, frame = cap.read()
 if ret and frame is not None:
 latest_frame = frame

 # Show live window
 cv2.imshow("Tello EDU Flight View", frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
 keep_running = False
 break
 else:
 time.sleep(0.01)

 cap.release()
 cv2.destroyAllWindows()

def save_snapshot(waypoint_label):
 global latest_frame
 os.makedirs('snapshots', exist_ok=True)
 if latest_frame is not None:
 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 filename = f"snap_{waypoint_label}_{timestamp}.jpg"
 path = os.path.join('snapshots', filename)
 cv2.imwrite(path, latest_frame)
 print(f"[SNAPSHOT] Captured Waypoint Photo [{waypoint_label}]: {path}")
 else:
 print(f"[WARNING] Video frame not available yet for waypoint [{waypoint_label}]")

def main():
 global keep_running
 print("=== Tello EDU Flight with Camera Stream ===")

 drone = Tello()

 try:
 # 1. Start SDK mode & turn Camera ON
 drone.send_command('command')
 drone.stream_on()
 time.sleep(1.5)

 # 2. Launch Camera Stream Worker Thread
 video_url = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
 stream_thread = threading.Thread(target=video_stream_worker, args=(video_url,))
 stream_thread.daemon = True
 stream_thread.start()

 # Give stream time to display window
 time.sleep(2.0)

 print("\n Commencing Flight Sequence with Camera Logging...")

 # Waypoint 1: Takeoff & Hover
 drone.send_command('takeoff')
 time.sleep(3.0)
 save_snapshot("1_hover")

 # Waypoint 2: Left 50cm
 drone.send_command('left 50')
 time.sleep(2.0)
 save_snapshot("2_left")

 # Waypoint 3: Right 50cm
 drone.send_command('right 50')
 time.sleep(2.0)
 save_snapshot("3_right")

 # Waypoint 4: Up 50cm
 drone.send_command('up 50')
 time.sleep(2.0)
 save_snapshot("4_up")

 # Waypoint 5: Down 50cm
 drone.send_command('down 50')
 time.sleep(2.0)
 save_snapshot("5_down")

 # Waypoint 6: Land
 drone.send_command('land')
 time.sleep(2.0)
 save_snapshot("6_landed")

 except KeyboardInterrupt:
 print("\n!! Emergency Stop: Landing drone...")
 drone.send_command('land')

 finally:
 keep_running = False
 print("Disabling video stream...")
 drone.stream_off()
 drone.close()
 print("=== Flight with Camera Stream Complete ===")

if __name__ == '__main__':
 main()
