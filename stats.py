from datetime import datetime

class Stats(object):
    """
    Tracks and logs command requests, responses, timestamps, and durations.
    """
    def __init__(self, command, id):
        self.command = command
        self.response = None
        self.id = id
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None

    def add_response(self, response):
        self.response = response
        self.end_time = datetime.now()
        self.duration = self.get_duration()

    def get_duration(self):
        if self.end_time and self.start_time:
            diff = self.end_time - self.start_time
            return diff.total_seconds()
        return 0

    def print_stats(self):
        print(f'\n[Stat ID {self.id}]')
        print(f'  Command   : {self.command}')
        print(f'  Response  : {self.response}')
        print(f'  Start Time: {self.start_time.strftime("%H:%M:%S.%f")[:-3]}')
        print(f'  End Time  : {self.end_time.strftime("%H:%M:%S.%f")[:-3] if self.end_time else "N/A"}')
        print(f'  Duration  : {self.duration} sec')

    def got_response(self):
        return self.response is not None

    def return_stats(self):
        res_str = f'\n--- Stat ID: {self.id} ---\n'
        res_str += f'Command   : {self.command}\n'
        res_str += f'Response  : {self.response}\n'
        res_str += f'Start Time: {self.start_time}\n'
        res_str += f'End Time  : {self.end_time}\n'
        res_str += f'Duration  : {self.duration} s\n'
        return res_str
