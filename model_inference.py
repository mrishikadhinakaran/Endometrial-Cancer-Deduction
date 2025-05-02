import tensorflow as tf
import numpy as np
import cv2
from src.llm_alternative import get_llm_explanation

# Define a custom InputLayer that ignores the 'batch_shape' argument.
from tensorflow.keras.layers import InputLayer as OriginalInputLayer

class CustomInputLayer(OriginalInputLayer):
    def __init__(self, *args, **kwargs):
        # Remove 'batch_shape' if it exists
        kwargs.pop("batch_shape", None)
        super().__init__(*args, **kwargs)

# Path to your complete saved model (architecture + weights)
MODEL_PATH = "models/im4MEC_model.keras"

# Load the model using the custom input layer to bypass the batch_shape issue.
try:
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False, 
        custom_objects={"InputLayer": CustomInputLayer}
    )
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit(1)

def preprocess_image(image_path):
    """Load and preprocess an image for model inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image at path: " + image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_with_model(image_path):
    """Run inference using the loaded model."""
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

def predict_with_explanation(image_path):
    """Perform inference and generate an explanation using the LLM integration."""
    predicted_class, confidence = predict_with_model(image_path)
    # Map numeric classes to human-readable labels (adjust mapping as needed)
    class_labels = {0: "Non-Cancerous", 1: "Cancerous"}
    label = class_labels.get(predicted_class, "Unknown")
    
    # Call the LLM integration with both label and confidence
    explanation = get_llm_explanation(label, confidence)
    
    # Return the actual values instead of placeholders.
    return {
        "predicted_class": label,
        "confidence": confidence,
        "explanation": explanation
    }

if __name__ == "__main__":
    # Update this path to a valid test image from your dataset.
    test_image_path = "processed_data/EA/1509779.JPG"
    result = predict_with_explanation(test_image_path)
    print(result)
