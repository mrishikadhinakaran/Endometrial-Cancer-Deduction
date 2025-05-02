import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define target size (based on model input)
TARGET_SIZE = (224, 224)

def preprocess_image(image_input):
    """
    Preprocess an image for model prediction.
    Handles both NumPy arrays (Gradio) and file paths.
    """
    try:
        # Check if input is already a NumPy array (Gradio input)
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype("uint8"))  # Convert array to PIL image
        else:
            image = Image.open(image_input).convert("RGB")  # Load from file path

        # Resize the image
        image = image.resize(TARGET_SIZE)

        # Convert image to NumPy array and normalize
        image_array = np.array(image) / 255.0  

        # Expand dimensions to match model input shape
        image_array = np.expand_dims(image_array, axis=0)

        # Display and save the image to check preprocessing
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title("Preprocessed Image")
        plt.axis("off")
        plt.savefig("preprocessed_image.png", bbox_inches="tight")  # Save image
        print("✅ Preprocessed image saved as 'preprocessed_image.png'")

        print(f"✅ Image processed successfully! Shape: {image_array.shape}")
        return image_array

    except Exception as e:
        print(f"❌ Error in image preprocessing: {e}")
        return None
