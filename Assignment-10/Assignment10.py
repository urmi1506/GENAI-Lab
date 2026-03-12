

from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion v1.5 model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate an image from a text prompt
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Example usage
generate_image("A boy riding a horse")