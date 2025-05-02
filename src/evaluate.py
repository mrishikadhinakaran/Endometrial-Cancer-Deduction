import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define Paths
model_path = "models/im4MEC_model.h5" 
val_dir = "dataset/val" 

# Check if the model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")

# Load the trained model
model = load_model(model_path)
print("Model loaded successfully!")

# Check if validation directory exists and is not empty
if not os.path.exists(val_dir) or not os.listdir(val_dir):
    raise FileNotFoundError(f"Validation dataset not found or empty at: {val_dir}")

# Data generator for validation images
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),  
    batch_size=32,
    class_mode='categorical'
)

# Ensure dataset is not empty
if val_generator.samples == 0:
    raise ValueError("Validation dataset is empty. Please check your data.")

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"\n🎯 Validation Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Validation Loss: {loss:.4f}")
