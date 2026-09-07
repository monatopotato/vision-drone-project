"""
Tello EDU Camera Stream & Snapshot Controller
Enables video streaming from Tello drone over UDP port 11111, displays live camera feed,
allows capturing snapshots, and toggling video stream ON/OFF.

Controls:
  's' - Save snapshot image to snapshots/ folder
  't' - Toggle camera stream ON / OFF
  'q' or ESC - Quit viewer and turn camera stream OFF
"""

import os
import cv2
import time
from datetime import datetime
from tello import Tello

def main():
    os.makedirs('snapshots', exist_ok=True)

    print("=== Tello EDU Camera Stream Viewer ===")
    print("Connecting to Tello EDU drone...")

    drone = Tello()
    
    # 1. Initialize SDK command mode
    if not drone.send_command('command'):
        print("!! Failed to connect to Tello SDK command mode. Check Wi-Fi connection.")
        drone.close()
        return

    # 2. Turn Camera Stream ON
    drone.stream_on()
    time.sleep(1.0) # Short delay to let H.264 stream start

    # 3. Open UDP video capture stream via OpenCV
    video_url = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
    print(f">> Opening video stream at '{video_url}'...")
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    is_streaming = True
    print("\n[CAMERA ONLINE] Controls: 's' = Save Snapshot | 't' = Toggle Stream | 'q' = Quit")

    try:
        while True:
            if is_streaming:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Draw HUD / UI Overlay
                    hud_frame = frame.copy()
                    cv2.putText(hud_frame, "TELLO EDU - CAMERA STREAM ONLINE", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(hud_frame, "[S] Snapshot  [T] Toggle  [Q] Quit", (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow("Tello EDU Camera Viewer", hud_frame)
                else:
                    # Frame drop or buffering warning
                    time.sleep(0.01)

            key = cv2.waitKey(1) & 0xFF
            
            # Press 'q' or ESC to Quit
            if key == ord('q') or key == 27:
                print("\nExiting viewer...")
                break

            # Press 's' to save snapshot
            elif key == ord('s') and is_streaming:
                if 'frame' in locals() and frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join('snapshots', f'tello_snap_{timestamp}.jpg')
                    cv2.imwrite(snap_path, frame)
                    print(f"[SNAPSHOT] Saved: {snap_path}")

            # Press 't' to toggle stream ON/OFF
            elif key == ord('t'):
                if is_streaming:
                    print("\n>> Turning camera stream OFF...")
                    drone.stream_off()
                    is_streaming = False
                    cap.release()
                else:
                    print("\n>> Turning camera stream ON...")
                    drone.stream_on()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
                    is_streaming = True

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected.")

    finally:
        # Cleanup & Turn Camera OFF
        print("Cleaning up and disabling video stream...")
        cap.release()
        cv2.destroyAllWindows()
        drone.stream_off()
        drone.close()
        print("=== Camera Stream Controller Closed ===")

if __name__ == '__main__':
    main()
