# Bibliotheken laden
from machine import Pin, PWM

PIN_RED = 14
PIN_GREEN = 15
PIN_BLUE = 17
PIN_IR = 18


# GPIO25 mit PWM initialisieren (Onboard-LED)
pwm_red = PWM(Pin(PIN_RED))
# Frequenz in Hertz (Hz) einstellen
pwm_red.freq(8)
# Tastgrad (Duty Cycle) einstellen
pwm_red.duty_u16(1000)

# GPIO25 mit PWM initialisieren (Onboard-LED)
pwm_green = PWM(Pin(PIN_GREEN))
# Frequenz in Hertz (Hz) einstellen
pwm_green.freq(8)
# Tastgrad (Duty Cycle) einstellen
pwm_green.duty_u16(1000)

# GPIO25 mit PWM initialisieren (Onboard-LED)
pwm_blue = PWM(Pin(PIN_BLUE))
# Frequenz in Hertz (Hz) einstellen
pwm_blue.freq(8)
# Tastgrad (Duty Cycle) einstellen
pwm_blue.duty_u16(1000)


# GPIO25 mit PWM initialisieren (Onboard-LED)
pwm_ir = PWM(Pin(PIN_IR))
# Frequenz in Hertz (Hz) einstellen
pwm_ir.freq(8)
# Tastgrad (Duty Cycle) einstellen
pwm_ir.duty_u16(1000)