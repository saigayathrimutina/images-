import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Use GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    safety_checker=None
).to(device)

# Function to generate image
def generate_image(prompt):
    return pipe(prompt, guidance_scale=7.5).images[0]

# Gradio interface
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter prompt..."),
    outputs=gr.Image(type="pil"),
    title="AI Image Generator"
).launch()
