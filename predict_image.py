import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

# Settings 
MODEL_PATH = "fruit_mobilenetv2_model.h5"  
CLASS_INDEX_PATH = "class_indices.json"     
IMG_SIZE = (224, 224)                       # MobileNetV2 input size

# Load model 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded successfully!")

# Load class mapping
if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
else:
    class_names = {0: "Fresh", 1: "Rotten"}  # fallback

print("Loaded classes:", class_names)

#Prediction function
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    # Load & preprocess
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    preds = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    print(f"\nPrediction for {os.path.basename(img_path)}:")
    print(f"   → {class_names[class_idx]} ({conf*100:.2f}% confidence)")


if __name__ == "__main__":
    test_img = "images-25-_jpeg.rf.c70cb6d11ea5330992536bf8f4a59a2f.jpg" #image input 
    predict_image(test_img)
