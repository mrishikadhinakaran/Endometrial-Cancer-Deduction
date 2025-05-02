import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load trained model
MODEL_PATH = "models/im4MEC_model.keras"
model = keras.models.load_model(MODEL_PATH)

# Image path (change this to your test image)
IMAGE_PATH = "dataset/sample.jpg"
IMG_SIZE = (224, 224)

# Preprocess image
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Output result
print(f"Predicted Class: {predicted_class}")
