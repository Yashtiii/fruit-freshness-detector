import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 32
image_size = 224
validation_split = 0.2

train_data = tf.keras.utils.image_dataset_from_directory(
r"D:\Study stuff\Fruit insights project\Fresh-Insight\Fresh-Insight\Training",
validation_split = validation_split,
subset = "training",
seed = 123,
image_size = (image_size, image_size),
batch_size = batch_size,
label_mode = "categorical",
shuffle = True,
)

validation_data = tf.keras.utils.image_dataset_from_directory(
r"D:\Study stuff\Fruit insights project\Fresh-Insight\Fresh-Insight\Training",
validation_split = validation_split,
subset = "validation",
seed = 123,
image_size = (image_size, image_size),
batch_size = batch_size,
label_mode = "categorical",
shuffle = True,
)

test_data = tf.keras.utils.image_dataset_from_directory(
r"D:\Study stuff\Fruit insights project\Fresh-Insight\Fresh-Insight\Testing",
image_size = (image_size, image_size),
batch_size = batch_size,
label_mode = "categorical",
shuffle = False,
)

print("Class Names: ", train_data.class_names)
class_names = train_data.class_names



#  MobileNetV2 Base Model
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # pehle sirf top layers train hongi

# Custom Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

#  Compile 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train 
history = model.fit(
    train_generator,
    epochs=10,   # pehle 10-15 epochs 
    validation_data=valid_generator
)

#  Evaluate
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy: {acc*100:.2f}%")

# Save 
model.save("fruit_mobilenetv2_model.h5")
print("💾 Model saved as fruit_mobilenetv2_model.h5")

# Plots 
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
