from picamera2 import Picamera2
import numpy as np
import mmap
import os
import time
import re
import cv2

import ioexpander as io
# GPIO replaced by ioexpander
#import RPi.GPIO as GPIO

# framebuffer device
FB_DEV = '/dev/fb0'
FRAME_TIME = 1/24

# Screen resolution
WIDTH = 800
HEIGHT = 480
BPP = 32  # bits per pixel

PIN = 14

SAVE_DIR = "/opt/imgs"
NAME_FORMAT = "IM_{}.jpg"
NAME_REGEX = r"IM_(\d*)\.jpg"

# Calculate the screen size in bytes
screensize = WIDTH * HEIGHT * (BPP // 8)

def image_to_565_hex_array(image):
    # Ensure the image is in the correct dtype
    image = image.astype(np.uint32)
    # rotate
    image = np.transpose(image, (1, 0, 2))
    
    # Use vectorized operations to convert RGB to a single hexadecimal value
    # take in the red, green and blue values (0-255) as 8 bit values and then combine
    # and shift them to make them a 16 bit hex value in 565 format. 
    # hex_array = (image[:, :, 0] << 16) + (image[:, :, 1] << 8) + image[:, :, 2]
    hex_array = ((image[:, :, 0] // 255 * 31 << 11) | (image[:, :, 1] << 8 // 255 * 63) << 5) | image[:, :, 2] // 255 * 31

    return hex_array

def main():
    os.putenv('SDL_FBDEV', FB_DEV)
    ioe = io.IOE(i2c_addr=0x18)
    ioe.set_mode(PIN, io.PIN_MODE_IO)

    # GPIO replaced by ioexpander
    # GPIO.setmode(GPIO.BOARD)
    # GPIO.setup(PIN, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (WIDTH, HEIGHT), 'format': 'BGR888'})
    cam.set_controls({"AwbEnable": False})
    cam.configure(config)
    cam.start()

    os.makedirs(SAVE_DIR, exist_ok=True)
    img_list = os.listdir(SAVE_DIR)
    if len(img_list) == 0:
        last_index = 1
    else:
        last_img = img_list[-1]
        last_index = int(re.match(NAME_REGEX, last_img).group(1))
    
    fb = np.memmap('/dev/fb0', dtype='uint16',mode='w+', shape=(WIDTH,HEIGHT))
    # fb[0:100, :] = 0xff0000
    # fb[100:200, :] = 0x00ff00
    # fb[200:300, :] = 0x0000ff
    # fb[300:400, :] = 0x313131


    while True:
        start = time.time()
        frame = cam.capture_array()
        print("Before", frame[0,0])
        fb[:] = image_to_565_hex_array(frame)
        sleep = FRAME_TIME - (time.time() - start)
        if ioe.input(14) == io.HIGH:
            last_index += 1
            name = os.path.join(SAVE_DIR, NAME_FORMAT.format(last_index))
            cam.capture_file(name)
            print("Capturing image", name)
        if sleep > 0:
            time.sleep(sleep)
            #print("Lagging behind {} seconds".format(-sleep))


if __name__ == '__main__':
    main()