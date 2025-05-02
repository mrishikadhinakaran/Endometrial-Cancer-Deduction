import torch
import torch.nn as nn
import torch.optim as optim
from model import CancerClassifier  # Ensure your model.py defines CancerClassifier

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters (adjust these values according to your actual model)
input_size = 512   # Example: adjust based on your model's expected input features
hidden_size = 256  # Example hidden layer size
output_size = 2    # Binary classification: Cancerous vs. Non-Cancerous

# Initialize the model
model = CancerClassifier(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training data (replace this with your actual data loading and preprocessing)
# For demonstration, we'll use 100 random samples
X_dummy = torch.randn(100, input_size).to(device)  # Dummy input data
y_dummy = torch.randint(0, output_size, (100,)).to(device)  # Dummy target labels

# Training loop (dummy training)
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_dummy)
    loss = criterion(outputs, y_dummy)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save the trained model's state dictionary as 'model.pth'
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
