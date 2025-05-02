import gradio as gr
import requests

def classify_cancer(image):
    url = "http://127.0.0.1:8000/predict/"
    files = {"file": image}
    response = requests.post(url, files=files)
    return response.json()["prediction"]

gr.Interface(fn=classify_cancer, inputs="image", outputs="text").launch()
