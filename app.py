import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Device setup: GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Function to generate image from text prompt
def generate_image(prompt):
    image = pipe(prompt, guidance_scale=7.5).images[0]
    return image

# Build Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=gr.Image(type="pil"),
    title="AI Image Generator",
    description="Generate images from text prompts using Stable Diffusion"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
