from transformers import pipeline

def download_gpt2():
    print("Downloading distilgpt2 model...")
    # Initialize the text generation pipeline with the distilgpt2 model.
    generator = pipeline("text-generation", model="distilgpt2", device=-1)
    # Generate a small text snippet to force the download.
    _ = generator("Hello", max_length=10)
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_gpt2()
