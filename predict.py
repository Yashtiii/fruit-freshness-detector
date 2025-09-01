import cv2 
import tensorflow as tf
import numpy as np
import serial
import time
import re
import json
import os
from collections import deque


MODEL_PATH = "fruit_mobilenetv2_model.h5"
CLASS_INDEX_PATH = "class_indices.json"
SERIAL_PORT = "COM5"     # apna COM port daalo
BAUD = 115200
CONF_THRESH = 0.70
FRAME_SIZE = (224, 224)

# Load model 
model = tf.keras.models.load_model(MODEL_PATH)

#  Load class mapping 
if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
else:
    class_names = {0: "Fresh", 1: "Rotten"}  # fallback

print("Loaded classes:", class_names)

#Serial (ESP32/Arduino) 
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
time.sleep(2)
last_sensor_line = "Waiting for sensor..."

#  Camera 
cap = cv2.VideoCapture(0)

#  Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#  Prediction Stabilization 
N = 15   # last 15 frames ka average lenge
pred_history = deque(maxlen=N)

def parse_temp_hum(line: str):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if len(nums) >= 2:
        hum_val = float(nums[0])
        temp_val = float(nums[1])
        return temp_val, hum_val
    return None, None

def survival_days(label: str, temp_val: float, hum_val: float):
    if label.lower() == "rotten":
        return " Rotten: <1 day left"
    if temp_val is None or hum_val is None:
        return " Fresh: conditions unknown"
    if temp_val > 30:
        return " Fresh: 1-2 days"
    if 20 <= temp_val <= 30 and 50 <= hum_val <= 80:
        return " Fresh: 3-4 days"
    if temp_val < 20 and 55 <= hum_val <= 65:
        return " Fresh: 5-7 days"
    return " Fresh: uncertain"

while True:
    # read sensor data 
    try:
        line = arduino.readline().decode(errors="ignore").strip()
        if line:
            if line.startswith("Humidity") or line.startswith("Failed"):
                last_sensor_line = line
    except:
        pass

    #  camera frame 
    ret, frame = cap.read()
    if not ret:
        print(" Camera not detected")
        break

    #  human detection 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        cv2.putText(frame, "Not a fruit (person detected)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, last_sensor_line, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Fruit Freshness + Sensor Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    #for prediction
    img = cv2.resize(frame, FRAME_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    class_idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    if conf < CONF_THRESH:
        label = "Not a fruit"
        class_idx = -1
    else:
        label = class_names[class_idx]


    pred_history.append((class_idx, conf))

    #  stabilize 
    if len(pred_history) > 0:
        classes, confs = zip(*pred_history)
        valid_classes = [cls for cls in classes if cls != -1]

        if len(valid_classes) > 0:
            final_class = max(set(valid_classes), key=valid_classes.count)
            avg_conf = np.mean([c for cls, c in pred_history if cls == final_class])
            label = class_names[final_class]
        else:
            final_class = -1
            avg_conf = np.mean(confs)
            label = "Not a fruit"
    else:
        final_class = class_idx
        avg_conf = conf

    temp_val, hum_val = parse_temp_hum(last_sensor_line)
    survival = survival_days(label, temp_val, hum_val)

    #draw on frame 
    cv2.putText(frame, f"{label} ({avg_conf*100:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, last_sensor_line, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, survival, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.imshow("Fruit Freshness + Sensor Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
