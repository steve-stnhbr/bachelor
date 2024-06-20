from picamera2 import Picamera2
import numpy as np
import mmap
import os
import time
import pygame
import re
import cv2

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


def main():
    os.putenv('SDL_FBDEV', FB_DEV)
    # Open the framebuffer device file
    # initialise PyGame engine
    pygame.init()
    pygame.mouse.set_visible(False)
    # set PyGame output to match dimensions of HyperPixel
    size = width, height = WIDTH, HEIGHT
    screen = pygame.display.set_mode(size)
    
    print("displaying test image")
    image = cv2.imread("test.jpg")
    image = cv2.resize(image, (800,600))
    surf = pygame.surfarray.make_surface(image)

    # This is the important bit
    def refresh():
        # We open the TFT screen's framebuffer as a binary file. Note that we will write bytes into it, hence the "wb" operator
        f = open("/dev/fb0","wb")
        # According to the TFT screen specs, it supports only 16bits pixels depth
        # Pygame surfaces use 24bits pixels depth by default, but the surface itself provides a very handy method to convert it.
        # once converted, we write the full byte buffer of the pygame surface into the TFT screen framebuffer like we would in a plain file:
        f.write(surf.convert(16,0).get_buffer())
        # We can then close our access to the framebuffer
        f.close()
        time.sleep(0.1)

    screen.blit(surf, (0, 0))
    pygame.display.update()
    refresh()

if __name__ == '__main__':
    main()