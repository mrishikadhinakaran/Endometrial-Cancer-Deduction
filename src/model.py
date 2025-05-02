import torch
import torch.nn as nn
import torch.optim as optim

class CancerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CancerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Example function to load model
def load_model(model_path="model.pth"):
    model = CancerClassifier(input_size=128, hidden_size=64, output_size=2)  # Modify as needed
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
