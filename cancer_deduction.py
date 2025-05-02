import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import AttentionNet  # ✅ Ensure this matches your architecture

# ✅ Load the trained model correctly
MODEL_PATH = "model.pth"  

# ✅ Initialize model with correct input size (matches preprocessed image tensor)
model = AttentionNet(model_size="small", input_feature_size=512 * 1024, n_classes=2)  # 🔥 Updated to 2 classes

# ✅ Load state dictionary properly
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model loaded successfully.")

# ✅ Define Correct Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((1024, 512)),  # 🔥 Ensure input size matches model expectations
    transforms.ToTensor(),
])

# ✅ Function for prediction
def predict(image):
    try:
        # Convert NumPy array to PIL Image
        img = Image.fromarray(image)  
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # 🔥 Flatten the input tensor before feeding to the model
        img = img.view(1, -1)  # Flatten (1, 512 * 1024)

        # Run model inference
        with torch.no_grad():
            logits, prob, _, _, _ = model(img)
            class_idx = torch.argmax(prob, dim=1).item()
        
        classes = ["Not Cancerous", "Cancerous"]  # 🔥 Updated class labels
        prediction = classes[class_idx]
        confidence = prob[0][class_idx].item()

        # ✅ Confidence Analysis
        if confidence < 0.6:
            confidence_note = "⚠️ Model is uncertain. Consider re-evaluating with more tests."
        elif confidence < 0.8:
            confidence_note = "🔍 Moderate confidence. Additional analysis may be helpful."
        else:
            confidence_note = "✅ High confidence in prediction."

        return f"{prediction} (Confidence: {confidence:.2f})", confidence_note

    except Exception as e:
        return f"Prediction Error: {str(e)}", "⚠️ Unable to analyze confidence."

# ✅ Gradio Interface (Simplified for Binary Classification)
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Endometrial Cancer Detection AI")

    with gr.Row():
        image_input = gr.Image(label="📷 Upload Histopathology Image")
        prediction_output = gr.Textbox(label="🔬 Prediction", interactive=False)
        confidence_output = gr.Textbox(label="📊 Confidence Analysis", interactive=False)

    classify_button = gr.Button("🔍 Classify")

    classify_button.click(predict, inputs=[image_input], outputs=[prediction_output, confidence_output])

demo.launch()
