"""
Simple Tello Drone Flight Script: Up 2 Seconds & Land
Takes off, ascends vertically for 2 seconds, descends back down, and lands safely.

Prerequisites:
- PC connected to Tello Wi-Fi (TELLO-XXXXXX)
- Battery level > 20%
- Open space away from obstacles
"""

import time
from tello import Tello

def fly_up_down():
    print("=== Tello Drone: Up 2 Seconds Flight Sequence ===")
    
    # 1. Initialize Tello Connection
    drone = Tello()
    
    try:
        # Enter SDK command mode
        if not drone.send_command('command'):
            print("!! Failed to communicate with Tello. Make sure you are connected to the drone's Wi-Fi network.")
            return

        print(">> Connected to Tello.")
        time.sleep(1)

        # 2. Takeoff
        print("\n>> 1. Taking off...")
        drone.send_command('takeoff')
        time.sleep(4)  # Wait for drone to complete takeoff and stabilize hover

        # 3. Move Up for 2 Seconds using RC Velocity Command
        print(">> 2. Ascending UP for 2 seconds...")
        drone.send_command('rc 0 0 35 0')  # throttle = 35 (ascend speed)
        time.sleep(2.0)

        # Stop vertical movement
        drone.send_command('rc 0 0 0 0')
        time.sleep(1)

        # 4. Move Down for 2 Seconds
        print(">> 3. Descending DOWN for 2 seconds...")
        drone.send_command('rc 0 0 -35 0')  # throttle = -35 (descend speed)
        time.sleep(2.0)

        # Stop movement & stabilize hover
        drone.send_command('rc 0 0 0 0')
        time.sleep(1)

        # 5. Land Safely
        print(">> 4. Landing...")
        drone.send_command('land')
        print(">> Flight Sequence Completed Successfully!")

    except KeyboardInterrupt:
        print("\n!! Emergency Interrupt! Initiating landing...")
        drone.send_command('rc 0 0 0 0')
        drone.send_command('land')

    except Exception as e:
        print(f"!! Error occurred: {e}")
        drone.send_command('rc 0 0 0 0')
        drone.send_command('land')

    finally:
        drone.close()
        print("=== Connection Closed ===")

if __name__ == '__main__':
    fly_up_down()
