import torch
import tensorflow as tf
import numpy as np
from models.model_architecture import CancerClassifier  
# Load the Keras model
keras_model = tf.keras.models.load_model("../models/cancer_classifier.keras")

# Convert Keras model to PyTorch
pytorch_model = CancerClassifier()
pytorch_model.load_state_dict(torch.load("..model.pth"))
pytorch_model.eval()

# Convert PyTorch model to TorchScript
dummy_input = torch.randn(1, 3, 224, 224)  
traced_model = torch.jit.trace(pytorch_model, dummy_input)

# Save 
traced_model.save("../triton/model_repository/cancer_model/1/model.pt")
print("Model converted and saved for Triton!")
