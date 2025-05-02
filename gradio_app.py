import gradio as gr
import tempfile
import os
from model_inference import predict_with_explanation

def gradio_predict(image):
    """
    This function receives a PIL image from Gradio, saves it temporarily,
    calls the inference function from model_inference.py, and returns the result.
    """
    # Save the PIL image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        image.save(temp_path)
    
    try:
        # Run the inference function (expects a file path)
        result = predict_with_explanation(temp_path)
    except Exception as e:
        result = {"predicted_class": "Error", "confidence": 0.0, "explanation": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Format the confidence as a percentage string
    confidence_str = f"{result['confidence'] * 100:.2f}%" if result["confidence"] is not None else "Error"
    
    return result["predicted_class"], confidence_str, result["explanation"]

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs=["text", "text", "text"],
    title="Molecular Endometrial Cancer Detection with Explanation",
    description="Upload a histopathological image to classify it as 'Cancerous' or 'Non-Cancerous' and receive an AI-generated explanation."
)

if __name__ == "__main__":
    # Listen on 0.0.0.0 so Docker can map the port to your host
    iface.launch(server_name="0.0.0.0")
