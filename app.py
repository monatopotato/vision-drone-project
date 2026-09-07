import os
import sys
import time
import argparse
from datetime import datetime
from tello import Tello

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Tello EDU Drone Flight Commander',
        epilog='Reference: https://tello.oneoffcoder.com/index.html'
    )
    parser.add_argument('-f', '--file', help='Path to command text file', default='command.txt')
    return parser.parse_args(args)

def start(file_name):
    if not os.path.exists(file_name):
        print(f"Error: Command file '{file_name}' not found.")
        sys.exit(1)

    print(f"=== Starting Flight Sequence from '{file_name}' ===")
    with open(file_name, 'r') as f:
        commands = f.readlines()

    os.makedirs('log', exist_ok=True)
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    tello = Tello()

    try:
        for line in commands:
            cmd = line.strip()
            if not cmd or cmd.startswith('#'):
                continue

            if cmd.startswith('delay'):
                try:
                    sec = float(cmd.partition('delay')[2].strip())
                    print(f"--> Hovering / Waiting for {sec} seconds...")
                    time.sleep(sec)
                except ValueError:
                    print(f"Invalid delay command format: '{cmd}'")
            else:
                tello.send_command(cmd)

    except KeyboardInterrupt:
        print("\n!! Emergency Stop: Keyboard Interrupt detected. Sending LAND command...")
        tello.send_command("land")

    finally:
        log_path = os.path.join('log', f'flight_log_{start_time_str}.txt')
        with open(log_path, 'w') as out:
            for stat in tello.get_log():
                stat.print_stats()
                out.write(stat.return_stats())
        print(f"\n=== Flight Sequence Complete. Log saved to '{log_path}' ===")
        tello.close()

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    start(args.file)
