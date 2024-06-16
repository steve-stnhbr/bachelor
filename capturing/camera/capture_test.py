from picamera2 import Picamera2
import cv2
# Screen resolution
WIDTH = 800
HEIGHT = 480
BPP = 32  # bits per pixel

def main():
    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (WIDTH, HEIGHT), 'format': 'BGR888'})
    cam.set_controls({"AwbEnable": False})
    cam.configure(config)
    cam.start()

    frame = cam.capture_array()
    cv2.imwrite("capture.jpg", frame)


if __name__ == '__main__':
    main()