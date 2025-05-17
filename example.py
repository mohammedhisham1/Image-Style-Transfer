#!/usr/bin/env python
# Example script for using the style transfer functionality

import os
import argparse
from PIL import Image
from diffusers.utils import load_image

from style_transfer import StyleTransfer

def main():
    """Example script for using the StyleTransfer class."""
    parser = argparse.ArgumentParser(description="Example of style transfer")
    
    parser.add_argument("--download", action="store_true", 
                        help="Download sample images first")
    parser.add_argument("--mode", type=str, default="basic", 
                        choices=["basic", "text", "variations", "blend"],
                        help="Style transfer mode")
    
    args = parser.parse_args()
    
    # Download sample images if requested
    if args.download or not os.path.exists("samples"):
        print("Downloading sample images...")
        import download_samples
        download_samples.main()
    
    # Initialize style transfer
    print("Initializing style transfer model...")
    transfer = StyleTransfer()
    
    # Process based on selected mode
    if args.mode == "basic":
        print("Performing basic style transfer...")
        
        # Load images
        content_image = load_image("samples/landscape.jpg")
        style_image = load_image("samples/vangogh.jpg")
        
        # Transfer style
        result = transfer.transfer_style(
            content_image,
            style_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Save and visualize
        result.save("output_basic.png")
        print(f"Result saved to output_basic.png")
        transfer.visualize_comparison(content_image, style_image, result)
        
    elif args.mode == "text":
        print("Performing text-guided style transfer...")
        
        # Load content image
        content_image = load_image("samples/portrait.jpg")
        
        # Text prompt
        style_prompt = "watercolor painting, soft colors, artistic, elegant, detailed brushstrokes"
        
        # Transfer style
        result = transfer.transfer_style_with_prompt(
            content_image,
            style_prompt,
            strength=0.7,
            guidance_scale=8.0,
            num_inference_steps=50
        )
        
        # Save result
        result.save("output_text.png")
        print(f"Result saved to output_text.png")
        
        # Visualize
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(content_image)
        axes[0].set_title("Content Image")
        axes[0].axis("off")
        
        axes[1].imshow(result)
        axes[1].set_title("Result")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
    elif args.mode == "variations":
        print("Generating style variations...")
        
        # Load images
        content_image = load_image("samples/city.jpg")
        style_image = load_image("samples/comic.jpg")
        
        # Generate variations
        variations = transfer.create_style_variations(
            content_image,
            style_image,
            variations=3,
            strength_range=(0.6, 0.8)
        )
        
        # Save variations
        for i, img in enumerate(variations):
            img.save(f"output_variation_{i+1}.png")
            print(f"Variation {i+1} saved to output_variation_{i+1}.png")
        
        # Visualize first variation
        transfer.visualize_comparison(content_image, style_image, variations[0])
        
    elif args.mode == "blend":
        print("Blending multiple styles...")
        
        # Load images
        content_image = load_image("samples/landscape.jpg")
        style1 = load_image("samples/vangogh.jpg")
        style2 = load_image("samples/watercolor.jpg")
        
        # Blend styles
        result = transfer.blend_styles(
            content_image,
            [style1, style2],
            weights=[0.7, 0.3],
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Save result
        result.save("output_blend.png")
        print(f"Result saved to output_blend.png")
        
        # Visualize
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(content_image)
        axes[0].set_title("Content")
        axes[0].axis("off")
        
        axes[1].imshow(style1)
        axes[1].set_title("Style 1 (70%)")
        axes[1].axis("off")
        
        axes[2].imshow(style2)
        axes[2].set_title("Style 2 (30%)")
        axes[2].axis("off")
        
        axes[3].imshow(result)
        axes[3].set_title("Blended Result")
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main() 