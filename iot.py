import serial
import time

# same port as Arduino IDE me show ho raha hai (abhi tumne bola COM5 hai)
arduino = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)  # connection stabilize

while True:
    if arduino.in_waiting > 0:   # check if data is available
        line = arduino.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print("From ESP32:", line)
