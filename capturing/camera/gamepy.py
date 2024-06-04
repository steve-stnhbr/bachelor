import pygame, os, time, sys, time

os.environ["SDL_FBDEV"] = "/dev/fb0"

# HyperPixel screen dimensions
WIDTH = 480
HEIGHT = 800

def main():
    # initialise PyGame engine
    pygame.init()
    pygame.mouse.set_visible(False)
    # set PyGame output to match dimensions of HyperPixel
    size = width, height = WIDTH, HEIGHT
    screen = pygame.display.set_mode(size)
    # clear screen (R,G,B) (255, 255, 255) = white background
    screen.fill((255, 255, 255))

    for i in range(0, width - 1):
        for j in range (0, height - 1):
            screen.set_at((i, j), (translate(i, 0, width - 1, 0, 255), translate(j, 0, height - 1, 0, 255), 0))

    pygame.display.flip()
    time.sleep(10)
    
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

if __name__ == '__main__':
    main()