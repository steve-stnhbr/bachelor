import RPi.GPIO as GPIO


def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(19, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

    while True:
        if GPIO.input(19) == GPIO.HIGH:
            print("Button")

if __name__ == '__main__':
    main()