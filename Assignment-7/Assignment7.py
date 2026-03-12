import os
from google import genai
from google.genai import types
from PIL import Image


client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def caption_image_directly(image_path):
    print("Analyzing image...")
    
    # Load the image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    
    prompt = """
    Perform a multimodal analysis of this image. 
    1. Identify the key objects and mood (like CLIP would).
    2. Write a professional, detailed caption based on those features.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Latest efficient vision model
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Run the system
image_file = "sample_image.png" 
final_caption = caption_image_directly(image_file)
print(f"\n--- Generated Caption ---\n{final_caption}")