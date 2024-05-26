import cv2
import socket
import pickle
import struct
import asyncio

async def handle_connection(client_socket, client_address, cap):
    print(f"Connection from {client_address} accepted")
    while True:
        ret, frame = cap.read()
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened")  
        exit(1)
    print("Server is listening...")
    try:
        while True:
            print("Waiting for connection...")
            client_socket, client_address = await loop.sock_accept(server_socket)
            print("Connection accepted")
            task = loop.create_task(handle_connection(client_socket, client_address, cap))
    except KeyboardInterrupt:
        print("Server is shutting down...")
        server_socket.close()
        cv2.destroyAllWindows()
        cap.release()

loop = asyncio.get_event_loop()
loop.create_task(start_server())
loop.run_forever()

