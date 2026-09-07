"""
Simple Single-File Tello EDU Flight Demo
Directly communicates with DJI / Ryze Tello EDU over UDP (Port 8889).

Flight Sequence:
1. Connect & enter SDK command mode ("command")
2. Takeoff and hover
3. Move Left 50 cm
4. Move Right 50 cm
5. Move Up 50 cm
6. Move Down 50 cm
7. Land back down
"""

import socket
import time

TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
LOCAL_PORT = 8889

def send_command(sock, command, delay=3):
 print(f">> Sending: {command}")
 sock.sendto(command.encode('utf-8'), (TELLO_IP, TELLO_PORT))

 # Simple response listener with timeout
 sock.settimeout(10.0)
 try:
 data, _ = sock.recvfrom(1024)
 print(f"<< Tello Response: {data.decode('utf-8').strip()}")
 except socket.timeout:
 print("!! Timed out waiting for response.")

 if delay > 0:
 print(f"Waiting {delay} seconds...")
 time.sleep(delay)

def main():
 # Bind local socket
 sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 sock.bind(('', LOCAL_PORT))

 print("=== Tello EDU Simple Flight Demo ===")
 print("Ensure you are connected to the Tello Wi-Fi network (TELLO-XXXXXX)\n")

 try:
 # Step 1: Initialize SDK command mode
 send_command(sock, 'command', delay=2)

 # Step 2: Takeoff & Hover
 send_command(sock, 'takeoff', delay=5)

 # Step 3: Move Left 50cm
 send_command(sock, 'left 50', delay=3)

 # Step 4: Move Right 50cm
 send_command(sock, 'right 50', delay=3)

 # Step 5: Move Up 50cm
 send_command(sock, 'up 50', delay=3)

 # Step 6: Move Down 50cm
 send_command(sock, 'down 50', delay=3)

 # Step 7: Land back down
 send_command(sock, 'land', delay=2)

 except KeyboardInterrupt:
 print("\n!! Keyboard Interrupt! Landing drone immediately...")
 send_command(sock, 'land', delay=1)

 finally:
 sock.close()
 print("=== Flight Demo Complete ===")

if __name__ == '__main__':
 main()
