"""
Simple Tello Drone Flight Script: Up 2 Seconds & Land
Takes off, ascends vertically for 2 seconds, descends back down, and lands safely.
"""

import time

try:
 from src.drone.tello import Tello
except ImportError:
 from tello import Tello

def fly_up_down():
 print("=== Tello Drone: Up 2 Seconds Flight Sequence ===")
 drone = Tello()

 try:
 if not drone.send_command('command'):
 print("!! Failed to communicate with Tello. Make sure you are connected to the drone's Wi-Fi network.")
 return

 print(">> Connected to Tello.")
 time.sleep(1)

 print("\n>> 1. Taking off...")
 drone.send_command('takeoff')
 time.sleep(4)

 print(">> 2. Ascending UP for 2 seconds...")
 drone.send_command('rc 0 0 35 0')
 time.sleep(2.0)

 drone.send_command('rc 0 0 0 0')
 time.sleep(1)

 print(">> 3. Descending DOWN for 2 seconds...")
 drone.send_command('rc 0 0 -35 0')
 time.sleep(2.0)

 drone.send_command('rc 0 0 0 0')
 time.sleep(1)

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
