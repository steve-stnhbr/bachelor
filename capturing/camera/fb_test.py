import numpy as np
import mmap
import os
import time


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

FB_DEV = '/dev/fb0'

# Screen resolution
width = 480
height = 800
bpp = 32  # bits per pixel

# Calculate the screen size in bytes
screensize = width * height * (bpp // 8)
# Open the framebuffer device file
with open(FB_DEV, 'r+b') as f:
    # Memory-map the framebuffer
    fb = mmap.mmap(f.fileno(), screensize, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
    
    # Create a numpy array with the size of the screen
    screen = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(0, width - 1):
        for j in range (0, height - 1):
            screen[j, i] = [translate(i, 0, width - 1, 0, 255), translate(j, 0, height - 1, 0, 255), 0]

    fb.write(screen.tobytes())
    fb.close()