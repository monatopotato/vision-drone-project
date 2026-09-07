"""
Interactive Text CLI Controller for Tello Drone
Allows typing natural commands to interactively control the drone in real-time.

Supported Commands:
  - 'takeoff' or 'start'   : Take off and hover
  - 'up'                   : Move UP 30 cm (or 'up 50' for custom distance)
  - 'down'                 : Move DOWN 30 cm (or 'down 50')
  - 'forward' or 'f'       : Move FORWARD 30 cm
  - 'back' or 'b'          : Move BACKWARD 30 cm
  - 'left' or 'l'          : Move LEFT 30 cm
  - 'right' or 'r'         : Move RIGHT 30 cm
  - 'turn left' / 'ccw'    : Rotate COUNTER-CLOCKWISE 45 deg
  - 'turn right' / 'cw'    : Rotate CLOCKWISE 45 deg
  - 'land'                 : Land drone safely
  - 'battery'              : Print current battery level
  - 'quit' or 'exit'       : Land and quit interface
"""

import sys
import time
from tello import Tello

DEFAULT_STEP_CM = 30

def print_help():
    print("""
=====================================================
          TELLO INTERACTIVE CLI CONTROLLER          
=====================================================
Available Text Commands:
  takeoff       : Take off and hover
  up [cm]       : Go UP 30 cm (e.g. 'up' or 'up 50')
  down [cm]     : Go DOWN 30 cm (e.g. 'down' or 'down 40')
  forward / f   : Move FORWARD 30 cm
  back / b      : Move BACKWARD 30 cm
  left / l      : Move LEFT 30 cm
  right / r     : Move RIGHT 30 cm
  turn left     : Rotate CCW 45 degrees
  turn right    : Rotate CW 45 degrees
  battery       : Check battery %
  land          : Land drone safely
  help          : Show this menu
  quit / exit   : Land and exit
=====================================================
""")

def process_command(drone, cmd_str):
    cmd = cmd_str.strip().lower()
    if not cmd:
        return True

    tokens = cmd.split()
    first_word = tokens[0]

    if first_word in ['help', 'h', '?']:
        print_help()

    elif first_word in ['takeoff', 'start', 'launch']:
        drone.send_command('takeoff')

    elif first_word in ['land', 'stop']:
        drone.send_command('land')

    elif first_word == 'up':
        dist = tokens[1] if len(tokens) > 1 and tokens[1].isdigit() else str(DEFAULT_STEP_CM)
        drone.send_command(f'up {dist}')

    elif first_word in ['down', 'dn']:
        dist = tokens[1] if len(tokens) > 1 and tokens[1].isdigit() else str(DEFAULT_STEP_CM)
        drone.send_command(f'down {dist}')

    elif first_word in ['forward', 'f', 'front']:
        dist = tokens[1] if len(tokens) > 1 and tokens[1].isdigit() else str(DEFAULT_STEP_CM)
        drone.send_command(f'forward {dist}')

    elif first_word in ['back', 'b', 'backward']:
        dist = tokens[1] if len(tokens) > 1 and tokens[1].isdigit() else str(DEFAULT_STEP_CM)
        drone.send_command(f'back {dist}')

    elif first_word in ['left', 'l']:
        dist = tokens[1] if len(tokens) > 1 and tokens[1].isdigit() else str(DEFAULT_STEP_CM)
        drone.send_command(f'left {dist}')

    elif first_word in ['right', 'r']:
        dist = tokens[1] if len(tokens) > 1 and tokens[1].isdigit() else str(DEFAULT_STEP_CM)
        drone.send_command(f'right {dist}')

    elif cmd in ['turn left', 'ccw']:
        drone.send_command('ccw 45')

    elif cmd in ['turn right', 'cw']:
        drone.send_command('cw 45')

    elif first_word in ['battery', 'bat']:
        drone.send_command('battery?')

    elif first_word in ['quit', 'exit', 'q']:
        print(">> Exiting controller... Landing drone for safety.")
        drone.send_command('land')
        return False

    else:
        # Fallback: Send raw SDK command if recognized
        print(f">> Sending custom raw SDK command: '{cmd}'")
        drone.send_command(cmd)

    return True

def main():
    print("=== Initializing Tello Text Interface ===")
    drone = Tello()
    
    if not drone.send_command('command'):
        print("!! Could not connect to Tello. Ensure your PC is connected to the drone's Wi-Fi network.")
        drone.close()
        return

    print(">> Connected successfully to Tello!")
    print_help()

    try:
        running = True
        while running:
            user_input = input("\n[Drone CLI] Enter command > ")
            running = process_command(drone, user_input)

    except KeyboardInterrupt:
        print("\n!! Emergency Keyboard Interrupt! Landing drone...")
        drone.send_command('land')

    finally:
        drone.close()
        print("=== Connection Closed ===")

if __name__ == '__main__':
    main()
