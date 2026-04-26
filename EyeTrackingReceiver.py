# EyeTrackingReceiver.py
import zmq

class EyeTrackingReceiver:
    def __init__(self, local_ip, remote_ip, port, use_remote, shared_data):
        self.lip = local_ip
        self.rip = remote_ip
        self.p = port
        self.status = use_remote
        self.BB = 0.0
        self.BE = 0.0
        self.eye_detected = True
        self.shared_data = shared_data

        # Initialize ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 0)

        # Connect to appropriate IP
        if self.status:
            self.socket.connect(f"tcp://{self.rip}:{self.p}")
        else:
            self.socket.connect(f"tcp://{self.lip}:{self.p}")
        print(
            f"Attempting to connect to {'remote' if self.status else 'local'} server at {'tcp://' + self.rip + ':' + str(self.p) if self.status else 'tcp://' + self.lip + ':' + str(self.p)}"
        )

    def receive_data(self):
        while True:
            if "stop" in self.shared_data and self.shared_data["stop"].value:
                print("EyeTrackingReceiver: Stop flag detected, exiting...")
                break

            last_event = None  # preserve latest non-empty event across the batch
            try:
                while True:  # Process all messages in the buffer
                    text = self.socket.recv_string(flags=zmq.NOBLOCK)
                    if text:
                        parsed_event = self.parse_data(text)
                        if parsed_event:
                            last_event = parsed_event
                        #print(text, end="\n")
            except zmq.error.Again:
                # Buffer drained — promote the last non-empty event seen
                if last_event is not None:
                    self.shared_data["eyeEvent"].value = last_event

            except KeyboardInterrupt:
                break

        self.socket.close()
        self.context.term()
        print("EyeTrackingReceiver stopped.")

    def parse_data(self, data_text):
        """Parse one message, update shared_data for all fields except eyeEvent,
        and return the event string (may be empty). The caller is responsible for
        writing the final eyeEvent so a non-empty event is not clobbered by a
        later empty-string message in the same drain batch."""
        data = data_text.split(";")
        # print(f"data: {data}")
        try:
            event = str(data[20])

            if data[20] == " NA":
                if self.eye_detected:
                    print("eyes are not detected")
                    self.eye_detected = False
            else:
                if not self.eye_detected:
                    print("eyes are detected, analyze start")
                    self.eye_detected = True
                self.shared_data["ID"].value = float(data[0])
                self.shared_data["Timestamp"].value = float(data[1])
                self.shared_data["PicNum"].value = int(data[11])
                self.shared_data["GazeX"].value = float(data[2])
                self.shared_data["GazeY"].value = float(data[3])
                self.shared_data["PupilSizeLeft"].value = float(data[4])
                self.shared_data["PupilSizeRight"].value = float(data[5])
                self.shared_data["RScore"].value = float(data[9])
                self.shared_data["LScore"].value = float(data[10])
                # eyeEvent is written by receive_data after the batch is drained
                return event

        except Exception as e:
            print(f"Error parsing data: {e}")
        return ""
