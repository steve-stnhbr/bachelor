import socket
import pickle
import struct
import asyncio
from picamera2 import Picamera2
import numpy as np

async def handle_connection(client_socket, client_address, cam):
    print(f"Connection from {client_address} accepted")
    while True:
        frame = cam.capture_array()
        frame_data = pickle.dumps(frame)
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

    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (640, 480), 'format': 'BGR888'}, lores={"size": (640, 480)}, display="lores")
    cam.set_controls({"AwbEnable": False})
    cam.configure(config)
    cam.start()
    try:
        while True:
            print("Waiting for connection...")
            client_socket, client_address = await loop.sock_accept(server_socket)
            print("Connection accepted")
            task = loop.create_task(handle_connection(client_socket, client_address, cam))
    except KeyboardInterrupt:
        print("Server is shutting down...")
        server_socket.close()

loop = asyncio.get_event_loop()
loop.create_task(start_server())
loop.run_forever()

