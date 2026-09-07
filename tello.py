import socket
import threading
import time
from stats import Stats

class Tello(object):
    """
    Abstract interface for communicating with Ryze / DJI Tello EDU via UDP sockets.
    Standard Tello IP: 192.168.10.1, Port: 8889
    """
    def __init__(self, tello_ip='192.168.10.1', tello_port=8889, local_port=8889):
        self.local_ip = ''
        self.local_port = local_port
        
        # Initialize UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.local_ip, self.local_port))

        self.tello_ip = tello_ip
        self.tello_port = tello_port
        self.tello_address = (self.tello_ip, self.tello_port)
        
        self.log = []
        self.MAX_TIME_OUT = 15.0

        # Background thread for receiving command ACKs
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def send_command(self, command):
        """
        Sends SDK command string to Tello UDP socket and waits for response.
        """
        stat = Stats(command, len(self.log))
        self.log.append(stat)

        print(f'>> Sending command: "{command}" to {self.tello_ip}:{self.tello_port}')
        self.socket.sendto(command.encode('utf-8'), self.tello_address)

        start = time.time()
        while not self.log[-1].got_response():
            now = time.time()
            if (now - start) > self.MAX_TIME_OUT:
                print(f'!! Timeout ({self.MAX_TIME_OUT}s) exceeded for command: "{command}"')
                return False
            time.sleep(0.1)

        print(f'<< Received response for "{command}": {self.log[-1].response}')
        return True

    def _receive_thread(self):
        """
        Continuously listens for incoming UDP response packets from Tello.
        """
        while True:
            try:
                data, ip = self.socket.recvfrom(1024)
                response = data.decode('utf-8', errors='ignore').strip()
                if self.log:
                    self.log[-1].add_response(response)
            except Exception as exc:
                print(f'!! Socket receive error: {exc}')
                break

    def stream_on(self):
        """
        Enables the video stream on Tello (stream UDP packets to port 11111).
        """
        print(">> Enabling video stream...")
        return self.send_command("streamon")

    def stream_off(self):
        """
        Disables the video stream on Tello.
        """
        print(">> Disabling video stream...")
        return self.send_command("streamoff")

    def get_log(self):
        return self.log

    def close(self):
        self.socket.close()

