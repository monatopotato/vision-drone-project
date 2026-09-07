"""
Autonomous AI Drone Assistant with Voice Commands & Terminal Natural Language Chat
Commands Examples (Voice or Typing):
  - "go to a ball and hover above it"
  - "once you go to a computer, land"
  - "find a bottle and hover"
  - "land" / "takeoff" / "exit"

Supports Voice Microphone Control, Terminal Chat Typing, and PC Webcam Simulation Mode.
"""

import os
import cv2
import time
import threading

try:
    from src.assistant.command_parser import parse_command
    from src.assistant.autonomous_tracker import AutonomousTracker
    from src.assistant.voice_listener import VoiceListener
    from src.detection.real_time_object_detector import RealTimeObjectDetector
    from src.drone.tello import Tello
except ImportError:
    from command_parser import parse_command
    from autonomous_tracker import AutonomousTracker
    from voice_listener import VoiceListener
    from real_time_object_detector import RealTimeObjectDetector
    from tello import Tello

# Global State
latest_frame = None
keep_running = True
drone = None
use_webcam = False
input_mode = "text"  # "text" or "voice"

def draw_assistant_hud(frame, hud_info, current_detections):
    if frame is None:
        return None

    hud = frame.copy()
    height, width, _ = hud.shape

    # 1. Top Banner Background
    cv2.rectangle(hud, (0, 0), (width, 50), (20, 20, 20), -1)
    
    state_str = hud_info.get('state', 'IDLE')
    target_str = hud_info.get('target', 'None')
    msg_str = hud_info.get('msg', '')

    mode_label = "VOICE MIC MODE" if input_mode == "voice" else "TERMINAL TEXT MODE"
    cv2.putText(hud, f"TELLO AI VISION ASSISTANT ({mode_label})", (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    color = (0, 255, 0) if state_str == 'REACHED' else (0, 215, 255) if state_str == 'TRACKING' else (200, 200, 200)
    status_txt = f"TARGET: {str(target_str).upper()} | STATE: {state_str} | {msg_str}"
    cv2.putText(hud, status_txt, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 2. Draw Target Reticle & Vector if object is locked
    box = hud_info.get('box')
    if box is not None:
        x1, y1, x2, y2 = box
        bx = int((x1 + x2) / 2.0)
        by = int((y1 + y2) / 2.0)
        cx = int(width / 2.0)
        cy = int(height / 2.0)

        cv2.rectangle(hud, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.arrowedLine(hud, (cx, cy), (bx, by), (0, 255, 255), 2, tipLength=0.2)
        cv2.circle(hud, (bx, by), 6, (0, 255, 0), -1)

    # Center Frame Crosshair
    cx, cy = int(width / 2.0), int(height / 2.0)
    cv2.line(hud, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
    cv2.line(hud, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

    # Bottom Instructions Bar
    cv2.rectangle(hud, (0, height - 25), (width, height), (15, 15, 15), -1)
    bottom_txt = "Voice Mode Active: Speak commands into mic" if input_mode == "voice" else "Terminal Typing Active: Type commands in terminal"
    cv2.putText(hud, f"{bottom_txt} | Press ESC to quit.",
                (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return hud

def execute_parsed_command(parsed, tracker, drone_inst):
    target = parsed['target']
    action = parsed['action']
    raw = parsed['raw_text']

    print(f"\n[AI Assistant] Processing instruction: \"{raw}\"")
    print(f"               --> Target: '{target}', Action: '{action}'")

    if action == 'TAKEOFF':
        if drone_inst:
            print(">> Launching takeoff...")
            drone_inst.send_command('takeoff')
        else:
            print(">> [SIMULATION] Takeoff executed.")

    elif action == 'LAND_IMMEDIATE':
        if drone_inst:
            print(">> Landing drone...")
            drone_inst.send_command('land')
        else:
            print(">> [SIMULATION] Landing executed.")

    elif action == 'UNKNOWN' or (target is None and action not in ['TAKEOFF', 'LAND_IMMEDIATE']):
        print("!! AI Assistant: I couldn't recognize a target object in your request.")
        print("   Available objects: ball, computer/laptop, bottle, cup, phone, chair, book, etc.")

    else:
        tracker.set_mission(target, action)
        print(f"[AI Assistant] Mission set! Navigating to '{target}' to execute '{action}'...")

def voice_command_thread(tracker, drone_inst):
    global keep_running
    listener = VoiceListener()
    if not listener.available:
        print("!! Voice recognition not available. Falling back to terminal text input.")
        terminal_chat_thread(tracker, drone_inst)
        return

    print("\n=======================================================")
    print("       VOICE-CONTROLLED DRONE ASSISTANT ONLINE         ")
    print("=======================================================")
    print("Speak natural instructions into your microphone. Examples:")
    print("  - \"go to a ball and hover above it\"")
    print("  - \"once you go to a computer, land\"")
    print("  - \"find a bottle and hover\"")
    print("  - \"takeoff\" / \"land\"")
    print("=======================================================\n")

    while keep_running:
        spoken_text = listener.listen_for_command(timeout=4, phrase_time_limit=6)
        if spoken_text:
            parsed = parse_command(spoken_text)
            execute_parsed_command(parsed, tracker, drone_inst)

def terminal_chat_thread(tracker, drone_inst):
    global keep_running
    print("\n=======================================================")
    print("       TELLO AI ASSISTANT TERMINAL CHAT ONLINE         ")
    print("=======================================================")
    print("Type your instructions below. Examples:")
    print("  > go to a ball and hover above it")
    print("  > once you go to a computer, land")
    print("  > find a bottle and hover")
    print("  > takeoff")
    print("  > land")
    print("=======================================================\n")

    while keep_running:
        try:
            user_input = input("\n[You] > ")
            if not user_input.strip():
                continue

            parsed = parse_command(user_input)
            execute_parsed_command(parsed, tracker, drone_inst)

        except (EOFError, KeyboardInterrupt):
            break

def video_stream_loop(video_source, detector, tracker, drone_inst):
    global latest_frame, keep_running
    cap = cv2.VideoCapture(video_source)
    
    while keep_running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        latest_frame = frame
        
        # 1. Run Object Detector
        ann_frame, detections = detector.detect_frame(frame)
        
        # 2. Compute Autonomous Control Step
        nav_cmd, hud_info = tracker.compute_control_step(detections, frame.shape[1], frame.shape[0])
        
        # 3. Send Flight Command if drone connected
        if nav_cmd and drone_inst and not use_webcam:
            if nav_cmd.startswith('rc') or nav_cmd in ['land', 'takeoff']:
                drone_inst.send_command(nav_cmd)

        # 4. Draw HUD
        hud = draw_assistant_hud(ann_frame if ann_frame is not None else frame, hud_info, detections)
        cv2.imshow("Tello AI Autonomous Assistant", hud)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            keep_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

def main(mode=None):
    global drone, use_webcam, keep_running, input_mode
    
    print("=== Launching Tello Vision-Guided Assistant ===")

    if mode is None:
        print("\nSelect Control Mode:")
        print("  [1] Voice Microphone Commands (Speak instructions)")
        print("  [2] Terminal Text Commands (Type instructions)")
        choice = input("Enter choice [1/2] (Default=1) > ").strip()
        input_mode = "voice" if choice != '2' else "text"
    else:
        input_mode = mode

    detector = RealTimeObjectDetector(confidence_threshold=0.40)
    tracker = AutonomousTracker()

    video_source = 0
    drone = Tello()
    
    if drone.send_command('command'):
        print(">> Connected to Tello Drone!")
        drone.stream_on()
        time.sleep(1.5)
        video_source = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
        use_webcam = False
    else:
        print("!! Could not connect to Tello Wi-Fi. Switching to WEBCAM SIMULATION MODE...")
        drone.close()
        drone = None
        use_webcam = True
        video_source = 0

    # Start selected input thread (Voice or Terminal Text)
    target_thread_fn = voice_command_thread if input_mode == "voice" else terminal_chat_thread
    c_thread = threading.Thread(target=target_thread_fn, args=(tracker, drone))
    c_thread.daemon = True
    c_thread.start()

    try:
        video_stream_loop(video_source, detector, tracker, drone)
    finally:
        keep_running = False
        if drone:
            drone.stream_off()
            drone.close()
        print("\n=== AI Drone Assistant Closed ===")

if __name__ == '__main__':
    main()
