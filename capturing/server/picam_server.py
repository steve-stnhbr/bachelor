import cv2
import socket
import pickle
import struct
import asyncio
from picamera2 import PiCamera
import numpy as np

async def handle_connection(client_socket, client_address, cam):
    print(f"Connection from {client_address} accepted")
    buf = np.empty((240, 320, 3), dtype=np.uint8)
    while True:
        cam.capture(buf, 'rgb')
        frame_data = pickle.dumps(buf)
        try:
            await asyncio.gather(
                loop.sock_sendall(client_socket, struct.pack("Q", len(frame_data))),
                loop.sock_sendall(client_socket, frame_data),
            )
            await asyncio.sleep(0.1)
        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected")
            break

async def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8888))
    server_socket.listen(5)
    server_socket.setblocking(False)

    cam = PiCamera()
    cam.resolution = (640, 480)
    cam.framerate = 4
    try:
        while True:
            print("Waiting for connection...")
            client_socket, client_address = await loop.sock_accept(server_socket)
            print("Connection accepted")
            task = loop.create_task(handle_connection(client_socket, client_address, cam))
    except KeyboardInterrupt:
        print("Server is shutting down...")
        server_socket.close()
        cv2.destroyAllWindows()

loop = asyncio.get_event_loop()
loop.create_task(start_server())
loop.run_forever()

