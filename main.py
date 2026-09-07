"""
Main Entry Point Launcher for Tello Vision-Guided Drone Project
Provides an interactive menu to launch AI Assistant, Object Detectors, CLI Controllers, or Trainers.
"""

import os
import sys

def main():
    print("""
======================================================
     VISION-GUIDED TELLO DRONE PROJECT LAUNCHER
======================================================
Select a mode to run:

  [1] Launch AI Vision Chat Assistant (Terminal Chat + Vision Tracking)
  [2] Launch Real-Time Object Bounding-Box Detector (PyTorch MobileNetV3 SSDLite)
  [3] Test Object Detector on PC Webcam (No Drone Required)
  [4] Launch Interactive Text CLI Drone Controller
  [5] Run Up-Down 2 Seconds Flight Test
  [6] Run Room Classification Model Evaluation
  [0] Exit
======================================================
""")
    choice = input("Enter choice [0-6] > ").strip()

    if choice == '1':
        from src.assistant.drone_assistant_chat import main as launch_assistant
        launch_assistant()
    elif choice == '2':
        from src.detection.drone_object_detector_ssd import main as launch_detector
        launch_detector()
    elif choice == '3':
        from src.detection.real_time_object_detector import run_webcam_detector
        run_webcam_detector()
    elif choice == '4':
        from src.cli.interactive_drone_cli import main as launch_cli
        launch_cli()
    elif choice == '5':
        from src.cli.up_down_flight import fly_up_down
        fly_up_down()
    elif choice == '6':
        from src.classification.eval_model import evaluate
        evaluate()
    elif choice == '0':
        print("Exiting project launcher. Goodbye!")
    else:
        print("Invalid choice. Please run python main.py again.")

if __name__ == '__main__':
    main()
