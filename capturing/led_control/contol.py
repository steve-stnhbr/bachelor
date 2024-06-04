
import RPi.GPIO as GPIO	       # Achtung: Schreibweise beachten: kleines i
import time

PIN_RED = 14
PIN_GREEN = 15
PIN_BLUE = 17
PIN_IR = 18

FREQ = 10
DU_CY = 100

def pwm(pin, frequency, duty_cycle):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin,GPIO.OUT)
    dimmer = GPIO.PWM(pin,frequency)
    dimmer.ChangeDutyCycle(duty_cycle)
    dimmer.start(0)

pwm(PIN_RED, FREQ, DU_CY)
pwm(PIN_GREEN, FREQ, DU_CY)
pwm(PIN_BLUE, FREQ, DU_CY)
pwm(PIN_IR, FREQ, DU_CY)