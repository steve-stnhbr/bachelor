import cv2
import socket
import pickle
import struct
from typing import Callable, Any
import asyncio

async def setup(image_callback: Callable[[cv2.typing.MatLike], Any]): 
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('pi.local', 8888))  # Replace 'server_ip_address' with the actual server IP
    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)  # 4K buffer size
            if not packet:
                break
            data += packet
        if not data:
            break
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        print('Message size:', msg_size)
        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)  # 4K buffer size
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        image_callback(frame)

if __name__ == '__main__':
    asyncio.run(setup(lambda frame: cv2.imshow('Client', frame)))
    cv2.destroyAllWindows()