

from diffusers import StableDiffusionPipeline
import torch

# Load the model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU

# Generate an image
prompt = "A cat eating an apple"
image = pipe(prompt).images[0]
image.show()

from IPython.display import display

display(image)