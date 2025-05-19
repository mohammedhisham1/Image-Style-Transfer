#!/usr/bin/env python
# Image Style Transfer using Stable Diffusion
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

class StyleTransfer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):

        self.model_id = model_id
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        self.pipe.enable_attention_slicing()
        
    def extract_style_prompt(self, reference_image, prompt_enhancer_strength=0.7):

        base_prompt = "high quality, detailed, elegant, masterpiece"
        
        ref_np = np.array(reference_image)
        
        colors = self._analyze_colors(ref_np)
        
        texture = self._analyze_texture(ref_np)
        
        enhanced_prompt = f"{base_prompt}, {colors}, {texture}"
        # ("high quality, detailed, elegant, masterpiece, vibrant colors, red tones, smooth texture")
        return enhanced_prompt
    


    def _analyze_colors(self, image_np):

        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        avg_s = np.mean(hsv[:,:,1])
        avg_v = np.mean(hsv[:,:,2])
        
        terms = []
        
        # Brightness
        if avg_v > 200:
            terms.append("bright lighting")
        elif avg_v < 100:
            terms.append("dark mood")
            
        # Saturation
        if avg_s > 150:
            terms.append("vibrant colors")
        elif avg_s < 70:
            terms.append("muted colors")
            
        hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        dominant_hue_index = np.argmax(hist)
        
        hue_terms = {
            0: "red tones",
            1: "orange tones", 
            2: "yellow tones",
            3: "yellow-green tones",
            4: "green tones",
            5: "turquoise tones",
            6: "cyan tones",
            7: "light blue tones",
            8: "blue tones",
            9: "purple tones",
            10: "magenta tones",
            11: "pink tones",
            12: "red tones",
            13: "orange tones",
            14: "yellow tones",
            15: "yellow-green tones", 
            16: "green tones",
            17: "turquoise tones"
        }
        
        terms.append(hue_terms.get(dominant_hue_index, ""))
        
        return ", ".join(terms)

    def _analyze_texture(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        avg_grad = np.mean(grad_mag)
        std_grad = np.std(grad_mag)
        
        terms = []
        
        # Edge prominence
        if avg_grad > 30:
            if std_grad > 50:
                terms.append("detailed linework")
            else:
                terms.append("strong lines")
        else:
            if std_grad > 20:
                terms.append("soft details")
            else:
                terms.append("smooth texture")
                
        # patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        high_freq = np.mean(magnitude[-20:, -20:])
        if high_freq > 5:
            terms.append("intricate patterns")
        
        return ", ".join(terms)

    def transfer_style(self, content_image, reference_image, strength=0.75, 
                      guidance_scale=7.5, num_inference_steps=50):
        
        content_image = self._preprocess_image(content_image)
        reference_image = self._preprocess_image(reference_image)

        style_prompt = self.extract_style_prompt(reference_image)
        print(f"Generated style prompt: {style_prompt}")
        
        mixed_image = Image.blend(content_image, reference_image, alpha=0.1)
        
        with torch.no_grad():
            images = self.pipe(
                prompt=style_prompt,
                image=content_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images
        
        return images[0]
    
    def transfer_style_with_prompt(self, content_image, prompt, strength=0.75,
                                  guidance_scale=7.5, num_inference_steps=50):
        
        content_image = self._preprocess_image(content_image)
        
        with torch.no_grad():
            images = self.pipe(
                prompt=prompt,
                image=content_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images
        
        return images[0]
    
    def _preprocess_image(self, image, target_size=512):

        # Keep aspect ratio
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
            
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create square image with black background
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        
        # Paste resized image in center
        new_img.paste(
            image, 
            ((target_size - new_width) // 2, (target_size - new_height) // 2)
        )
        
        return new_img
    
    def create_style_variations(self, content_image, reference_image, variations=3,
                              strength_range=(0.6, 0.8)):
        results = []
        
        style_prompt = self.extract_style_prompt(reference_image)
        print(f"Style prompt for variations: {style_prompt}")
        
        content_image = self._preprocess_image(content_image)
        
        min_strength, max_strength = strength_range
        for i in range(variations):

            strength = min_strength + (max_strength - min_strength) * (i / max(1, variations - 1))
            guidance = 7 + (i % 3 - 1)  # 6, 7, or 8
            steps = 50 + (i % 2) * 10     # 50 or 60
            
            print(f"Variation {i+1}: strength={strength:.2f}, guidance={guidance}")
            
            with torch.no_grad():
                images = self.pipe(
                    prompt=style_prompt,
                    image=content_image,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps
                ).images
                
            results.append(images[0])
            
        return results
    
    def blend_styles(self, content_image, reference_images, weights=None, strength=0.75,
                    guidance_scale=7.5, num_inference_steps=50):

        if weights is None:
            weights = [1.0 / len(reference_images)] * len(reference_images)
            
        if len(weights) != len(reference_images):
            raise ValueError("Number of weights must match number of reference images")
            
        if abs(sum(weights) - 1.0) > 1e-5:
            raise ValueError("Weights must sum to 1")
        
        content_image = self._preprocess_image(content_image)
        processed_refs = [self._preprocess_image(ref) for ref in reference_images]
        
        style_prompts = []
        for i, ref_img in enumerate(processed_refs):
            prompt = self.extract_style_prompt(ref_img)
            parts = prompt.split(", ")

            keep_count = max(1, int(len(parts) * weights[i]))
            selected_parts = parts[:keep_count]
            style_prompts.append(", ".join(selected_parts))
        
        final_prompt = ", ".join(style_prompts)
        print(f"Combined style prompt: {final_prompt}")
        

        with torch.no_grad():
            images = self.pipe(
                prompt=final_prompt,
                image=content_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images
            
        return images[0]
    
    def save_result(self, image, save_path):

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        image.save(save_path)
        print(f"Image saved to {save_path}")























































        
        

# def main():
#     parser = argparse.ArgumentParser(description="Image Style Transfer using Stable Diffusion")
    
#     parser.add_argument("--content", type=str, required=True, help="Path to content image")
#     parser.add_argument("--style", type=str, required=True, help="Path to style reference image")
#     parser.add_argument("--output", type=str, default="output.png", help="Output path")
#     parser.add_argument("--strength", type=float, default=0.75, help="Style transfer strength (0-1)")
#     parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
#     parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
#     parser.add_argument("--variations", type=int, default=0, help="Number of variations to generate")
#     parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", 
#                         help="Model ID to use")
#     parser.add_argument("--prompt", type=str, default=None, 
#                         help="Optional text prompt to guide style transfer")
#     parser.add_argument("--device", type=str, default=None, 
#                         help="Device to use (cuda or cpu)")
    
#     args = parser.parse_args()
    
#     # Initialize style transfer
#     transfer = StyleTransfer(model_id=args.model, device=args.device)
    
#     # Load content image
#     if not os.path.exists(args.content):
#         print(f"Content image not found: {args.content}")
#         return
#     content_image = load_image(args.content)
    
#     # Process based on arguments
#     if args.variations > 0:
#         # Load style image
#         style_image = load_image(args.style)
        
#         # Generate variations
#         results = transfer.create_style_variations(
#             content_image, 
#             style_image, 
#             variations=args.variations
#         )
        
#         # Save variations
#         output_dir = os.path.dirname(args.output)
#         filename = os.path.basename(args.output)
#         name, ext = os.path.splitext(filename)
        
#         for i, img in enumerate(results):
#             variation_path = os.path.join(output_dir, f"{name}_var{i+1}{ext}")
#             transfer.save_result(img, variation_path)
            

        
#     else:
#         # Standard style transfer
#         if args.prompt:
#             # Text-guided style transfer
#             result = transfer.transfer_style_with_prompt(
#                 content_image,
#                 args.prompt,
#                 strength=args.strength,
#                 guidance_scale=args.guidance,
#                 num_inference_steps=args.steps
#             )
            
#             # Visualize (without style reference)
#             fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#             axes[0].imshow(content_image)
#             axes[0].set_title("Content")
#             axes[0].axis("off")
#             axes[1].imshow(result)
#             axes[1].set_title("Result")
#             axes[1].axis("off")
#             plt.tight_layout()
#             plt.show()
            
#         else:
#             # Reference image style transfer
#             style_image = load_image(args.style)
            
#             result = transfer.transfer_style(
#                 content_image,
#                 style_image,
#                 strength=args.strength,
#                 guidance_scale=args.guidance,
#                 num_inference_steps=args.steps
#             )
            

        
#         # Save result
#         transfer.save_result(result, args.output)


# if __name__ == "__main__":
#     main() 