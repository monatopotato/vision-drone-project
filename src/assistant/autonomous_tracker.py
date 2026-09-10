"""
Autonomous Visual Servoing Tracker & Flight Engine for Tello Drone
Calculates centering vectors, distance approach, and flight navigation commands
to track target objects in real time.
"""

import time

class AutonomousTracker:
    def __init__(self, target_area_threshold=0.15, deadzone=0.12):
        self.target_object = None
        self.target_action = 'HOVER'  # 'HOVER' or 'LAND'
        self.state = 'IDLE'  # 'IDLE', 'SEARCHING', 'ALIGNING', 'APPROACHING', 'REACHED', 'LANDING'
        
        self.target_area_threshold = target_area_threshold  # Target occupies ~15% of frame when arrived
        self.deadzone = deadzone  # Centering tolerance deadzone
        self.search_start_time = 0
        self.arrival_time = 0

    def set_mission(self, target_object, action='HOVER'):
        self.target_object = target_object
        self.target_action = action
        self.state = 'SEARCHING' if target_object else 'IDLE'
        self.search_start_time = time.time()
        self.arrival_time = 0
        print(f"\n[MISSION UPDATED] Target Object: '{self.target_object}' | Action: '{self.target_action}' | State: {self.state}")

    def compute_control_step(self, detections, frame_width, frame_height):
        """
        Processes object detections in current frame and returns:
          nav_cmd: string Tello SDK command or RC parameters (e.g. 'rc 0 20 0 0', 'cw 30', 'land', etc.)
          hud_info: dict with telemetry details for HUD overlay
        """
        if self.state in ['IDLE', 'COMPLETED']:
            return None, {'state': self.state, 'target': self.target_object, 'box': None, 'msg': 'Standing by'}

        # Find best matching target bounding box
        target_box = None
        best_score = 0.0

        if self.target_object and detections:
            for det in detections:
                if det['label'] == self.target_object and det['score'] > best_score:
                    best_score = det['score']
                    target_box = det['box']

        # STATE 1: SEARCHING (rotate drone if object not in frame)
        if target_box is None:
            if self.state != 'SEARCHING':
                self.state = 'SEARCHING'
                self.search_start_time = time.time()

            # Rotate slowly to search
            nav_cmd = 'rc 0 0 0 25'  # Gentle CW rotation
            msg = f"Searching for '{self.target_object}' (rotating...)"
            return nav_cmd, {'state': 'SEARCHING', 'target': self.target_object, 'box': None, 'msg': msg}

        # Object is in frame! Calculate Centering & Distance
        x1, y1, x2, y2 = target_box
        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        box_area = (bw * bh) / float(frame_width * frame_height)

        # Centering errors normalized (-0.5 to +0.5)
        err_x = (bx - (frame_width / 2.0)) / float(frame_width)
        err_y = (by - (frame_height / 2.0)) / float(frame_height)

        # STATE 2: ALIGNING & APPROACHING
        self.state = 'TRACKING'

        rc_lr = 0
        rc_ud = 0
        rc_fb = 0
        rc_yaw = 0

        # 1. Align Yaw (Left / Right)
        if abs(err_x) > self.deadzone:
            rc_yaw = int(err_x * 50)
            rc_yaw = max(-35, min(35, rc_yaw))

        # 2. Align Altitude (Up / Down)
        if abs(err_y) > self.deadzone:
            rc_ud = int(-err_y * 50)
            rc_ud = max(-30, min(30, rc_ud))

        # 3. Forward Distance Approach
        if box_area < self.target_area_threshold:
            rc_fb = 25  # Move forward at moderate speed
            msg = f"Approaching '{self.target_object}' (Area: {box_area*100:.1f}%)"
        else:
            # Target Reached!
            self.state = 'REACHED'
            msg = f"TARGET REACHED! Executing {self.target_action}..."

        # If Target is Reached, execute post-arrival action
        if self.state == 'REACHED':
            if self.target_action == 'LAND':
                nav_cmd = 'land'
                self.state = 'COMPLETED'
            else:  # HOVER
                nav_cmd = 'rc 0 0 0 0'
                if self.arrival_time == 0:
                    self.arrival_time = time.time()
                msg = f"HOVERING above '{self.target_object}'"

        else:
            nav_cmd = f"rc {rc_lr} {rc_fb} {rc_ud} {rc_yaw}"

        hud_info = {
            'state': self.state,
            'target': self.target_object,
            'box': target_box,
            'err_x': err_x,
            'err_y': err_y,
            'area_pct': box_area * 100,
            'msg': msg
        }

        return nav_cmd, hud_info
