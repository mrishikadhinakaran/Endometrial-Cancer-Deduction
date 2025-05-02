from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2", device=-1)

def get_llm_explanation(prediction, confidence):
    """
    Generate a textual explanation using a lightweight GPT-2 model.
    
    Args:
        prediction (str): The predicted class label (e.g., "Cancerous").
        confidence (float): The model's confidence as a decimal (e.g., 0.60).
        
    Returns:
        explanation (str): Generated explanation text.
    """
    prompt = (f"Explain the significance and characteristics of histopathological images classified as "
              f"'{prediction}' in endometrial cancer detection, with a confidence of {confidence * 100:.2f}%. "
              "Provide key clinical insights and details.")
    
    # Generate explanation with truncation enabled to control max length.
    result = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)
    explanation = result[0]['generated_text']
    return explanation

if __name__ == "__main__":
    # Test the function with a sample prediction and confidence.
    test_explanation = get_llm_explanation("Cancerous", 0.60)
    print("Generated Explanation:", test_explanation)
