import tensorflow as tf
import numpy as np
import cv2
import sys

# Load the trained model
model_path = "models/im4MEC_model.h5"
try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Run inference
def predict(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return class_idx, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/infer.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        class_idx, confidence = predict(image_path)
        print(f"🧬 Predicted Class: {class_idx}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
