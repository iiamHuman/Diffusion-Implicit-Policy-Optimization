import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from PIL import Image
import os

# Configuration parameters
MODEL_PATH = "/home/wxd/GG/DiffusionDPO-main/tmp-sdxl-8.192e-9"
PROMPT = "a beautiful landscape with mountains and lake, highly detailed, 8k"
NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly"
OUTPUT_DIR = "./generated_images"
OUTPUT_FILENAME = "output.png"

NUM_INFERENCE_STEPS = 30  
ETA = 0.0  
SEED = 42  

# Image parameters
WIDTH = 1024
HEIGHT = 1024
GUIDANCE_SCALE = 7.5

def load_model(model_path):

    scheduler = DDIMScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1
    )
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
        pipeline.enable_attention_slicing()
    
    print(f"Model loaded successfully on {device}")
    return pipeline

def generate_image(pipeline, prompt, negative_prompt, steps, eta, seed, width, height, guidance_scale):
    """Generate image using DDIM sampling"""
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Steps: {steps}, Eta: {eta}, Seed: {seed}")
    
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    # Generate image
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            eta=eta,
            generator=generator,
            width=width,
            height=height,
            guidance_scale=guidance_scale
        ).images[0]
    
    return image

def save_image(image, output_dir, filename):
    """Save generated image to file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    return output_path

def main():
    """Main function to run the image generation process"""
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model path '{MODEL_PATH}' does not exist!")
            print("Please update the MODEL_PATH variable with your actual SDXL model path.")
            return
        
        print("Starting SDXL DDIM image generation...")
        print("=" * 50)
        
        pipeline = load_model(MODEL_PATH)
        
        image = generate_image(
            pipeline=pipeline,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            steps=NUM_INFERENCE_STEPS,
            eta=ETA,
            seed=SEED,
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=GUIDANCE_SCALE
        )
        
        # Save image
        output_path = save_image(image, OUTPUT_DIR, OUTPUT_FILENAME)
        
        print("=" * 50)
        print("Image generation completed successfully!")
        print(f"Generated image size: {image.size}")
        print(f"Output path: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your model path and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()