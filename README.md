# Fruit Freshness Detector

This project detects whether a fruit is **fresh or rotten** using a machine learning model.  
It also predicts the **survival days of the fruit** based on surrounding **temperature and humidity** using a **DHT11 sensor** with an **ESP32 microcontroller**.  
Additionally, the system can **detect humans or other objects** using object detection.


## Features
- Predicts fruit freshness (Fresh or Rotten)
- Shows accuracy percentage of the prediction
- Estimates survival days of the fruit based on environmental conditions
- Detects humans or other objects in the vicinity
- Works with ESP32 and DHT11 sensor



## SYSTEM REQUIRMENT-->

### Hardware
- ESP32 microcontroller
- DHT11 temperature and humidity sensor
- Camera (for object detection)

### Software
- Python 3.x
- OpenCV
- TensorFlow / Keras
- Other ML libraries as listed in `requirements.txt`



## Installation

1. Clone the repository:

   git clone https://github.com/Vishal-Bytee/fruit-freshness-detector.git
   cd fruit-freshness-detector
>python -m venv .venv
>source .venv/bin/activate   # For Windows: .venv\Scripts\activate
>pip install -r requirements.txt

USAGE-->

-->Connect the ESP32 and DHT11 sensor.

-->Run the main Python script:

-->python main.py


The system will:

* Predict fruit freshness
* Show accuracy percentage
* Display estimated survival days
* Detect humans or other objects if present


##DATASET-->

* Fruit quality dataset (Fresh and Rotten)
* Dataset is not included in the repository due to size.
* You can download it from [https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten?utm_source=chatgpt.com] and place it in the DATASET/ folder.

##FUTURE IMPROVEMENT-->

* Real-time monitoring with ESP32
* Improve model accuracy with more training data
* Deploy as a web or mobile application
