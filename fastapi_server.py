from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

app = FastAPI()

model = torch.load("best_model.pth", map_location=torch.device("cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")  # REMOVE TRAILING SLASH
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    
    return {"prediction": prediction}
