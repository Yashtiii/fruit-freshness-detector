import serial
import time


arduino = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)  

while True:
    if arduino.in_waiting > 0:  
        line = arduino.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print("From ESP32:", line)
