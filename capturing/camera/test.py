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
WIDTH = 640
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
    size = (0, 0)
    print("Display Info", pygame.display.Info())
    print("Display Modes", pygame.display.list_modes())
    print("Mode OK?", pygame.display.mode_ok(size))

    # drivers = ['fbcon', 'directfb', 'svgalib', 'x11']
    # found = False
    # for driver in drivers:
    #     # Make sure that SDL_VIDEODRIVER is set
    #     if not os.getenv('SDL_VIDEODRIVER'):
    #         os.putenv('SDL_VIDEODRIVER', driver)
    #     try:
    #         pygame.display.init()
    #     except pygame.error:
    #         print("Driver: {0} failed.".format(driver))
    #         continue
    #     found = True
    #     print("Found driver", driver)
    #     break

    # if not found:
    #     raise Exception('No suitable video driver found!')


    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()

    red = (255, 0, 0)
    screen.fill(red)
    # Update the display
    pygame.display.update()

    for _ in range(20):
        screen.fill((0, 0, 0))
        pygame.display.flip()
        clock.tick(1)
        screen.fill((200, 200, 0))
        pygame.display.flip()
        clock.tick(1)

if __name__ == '__main__':
    main()