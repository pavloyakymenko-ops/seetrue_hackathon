# main.py

import argparse
import multiprocessing
from EyeTrackingReceiver import EyeTrackingReceiver
from SceneImageReceiver import SceneImageReceiver
from multiprocessing import shared_memory
import numpy as np
import signal
import sys
from process import process


def main_process(shared_data, shm_name):
    print("enter main_process")
    # Reattach to shared memory in this process
    shm_local = shared_memory.SharedMemory(name=shm_name)
    image_buffer_scene = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm_local.buf)
    main = process(shared_data, image_buffer_scene)
    main.run()


# Eye Tracking Data Reception Setup
def eye_tracking_process(local_ip, remote_ip, port, use_remote, shared_data):
    print("enter eye_tracking_process")
    receiver = EyeTrackingReceiver(local_ip, remote_ip, port, use_remote, shared_data)
    receiver.receive_data()


#scene image handler and visualizer
def scene_image(local_ip, remote_ip, port, use_remote, shm_name, shared_data):
    print("enter scene_image")
    # Reattach to shared memory in this process
    shm_local = shared_memory.SharedMemory(name=shm_name)
    image_buffer_scene = np.ndarray((480, 640, 3), dtype=np.uint8, buffer=shm_local.buf)
    window = SceneImageReceiver(
        local_ip, remote_ip, port, use_remote, image_buffer_scene, shared_data
    )
    window.receive_data()


def signal_handler(sig, frame, shm):
    print(f"Get {sig} Signal, finishing process")
    shm.close()
    shm.unlink()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, shm))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, shm))

    manager = multiprocessing.Manager()
    shared_data = {
        "ID": manager.Value("d", 0.0),
        "Timestamp": manager.Value("d", 0.0),
        "PicNum": manager.Value("i", 0),
        "GazeX": manager.Value("d", 0.0),
        "GazeY": manager.Value("d", 0.0),
        "RScore": manager.Value("d", 0.0),
        "LScore": manager.Value("d", 0.0),
        "PupilSizeLeft": manager.Value("d", 0.0),
        "PupilSizeRight": manager.Value("d", 0.0),
        "eyeEvent": manager.Value("u", ""),
        "stop": manager.Value("b", False), 
    }

    max_image_size_scene = 640 * 480 * 3  # width * height * channels (BGR)
    shm = shared_memory.SharedMemory(create=True, size=max_image_size_scene)

    parser = argparse.ArgumentParser(
        description="IP addresses, port, and remote status for eye tracking data reception."
    )
    parser.add_argument(
        "--local_ip", type=str, default="127.0.0.1", help="Local IP address"
    )
    parser.add_argument(
        "--remote_ip", type=str, default="172.20.10.3", help="Remote IP address"
    )
    # parser.add_argument("--remote_ip", type=str, default="192.168.1.102", help="Remote IP address")
    parser.add_argument("--port", type=int, default=3428, help="Port number")
    parser.add_argument(
        "--use_remote", type=bool, default=False, help="Use remote IP? True/False"
    )

    args = parser.parse_args()

    # run the eye tracking data receiver
    process1 = multiprocessing.Process(
        target=eye_tracking_process,
        args=(args.local_ip, args.remote_ip, args.port, args.use_remote, shared_data),
    )
    #run the scene image receiver 
    process2 = multiprocessing.Process(
        target=scene_image,
        args=(args.local_ip, args.remote_ip, 3425, args.use_remote, shm.name, shared_data), 
    )
    #run the main process that processes the data and visualizes the gaze overlay
    process3 = multiprocessing.Process(
        target=main_process, args=(shared_data, shm.name)
    )

    process1.start()
    process2.start()
    process3.start()

    try:
        process1.join()
        process2.join()
        process3.join()
    finally:
        try:
            shm.close()
            shm.unlink()
        except Exception as e:
            print(f"Error cleaning up shared memory: {e}")
        print("All processes stopped.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
