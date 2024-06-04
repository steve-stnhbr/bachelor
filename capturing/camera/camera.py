from picamera2 import Picamera2
import numpy as np
import mmap
import os
import time
import pygame
import RPi.GPIO as GPIO

# framebuffer device
FB_DEV = '/dev/fb0'
FRAME_TIME = 1/24

# Screen resolution
WIDTH = 800
HEIGHT = 480
BPP = 32  # bits per pixel

PIN = 23

# Calculate the screen size in bytes
screensize = WIDTH * HEIGHT * (BPP // 8)


def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PIN, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (WIDTH, HEIGHT), 'format': 'BGR888'})
    cam.set_controls({"AwbEnable": False})
    cam.configure(config)
    cam.start()

    # Open the framebuffer device file
    # initialise PyGame engine
    pygame.init()
    pygame.mouse.set_visible(False)
    # set PyGame output to match dimensions of HyperPixel
    size = width, height = WIDTH, HEIGHT
    screen = pygame.display.set_mode(size)

    while True:
        start = time.time()
        frame = cam.capture_array()
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.update()
        sleep = FRAME_TIME - (time.time() - start)
        if GPIO.input(PIN) == GPIO.HIGH:
            print("Button")
        if sleep > 0:
            time.sleep(sleep)
            #print("Lagging behind {} seconds".format(-sleep))


if __name__ == '__main__':
    main()

