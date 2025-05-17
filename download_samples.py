#!/usr/bin/env python
# Utility script to download sample images for testing
import os
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO


def download_image(url, save_path):
    """Download an image from a URL and save it to the specified path.
    
    Args:
        url (str): URL of the image to download
        save_path (str): Path to save the image
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192), 
                             desc=f"Downloading {os.path.basename(save_path)}"):
                file.write(chunk)
                
        print(f"Downloaded {url} to {save_path}")
        
        # Verify the image can be opened
        img = Image.open(save_path)
        print(f"Image size: {img.size}")
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def main():
    # Create samples directory
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Sample content images
    content_images = [
        # Landscape photo
        ("https://images.unsplash.com/photo-1506744038136-46273834b3fb", "landscape.jpg"),
        # Portrait photo
        ("https://images.unsplash.com/photo-1544005313-94ddf0286df2", "portrait.jpg"),
        # City photo
        ("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df", "city.jpg"),
    ]
    
    # Sample style reference images
    style_images = [
        # Van Gogh style
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1920px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg", "vangogh.jpg"),
        # Watercolor style
        ("https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5", "watercolor.jpg"),
        # Comic book style
        ("https://i.imgur.com/XKHaZXr.jpg", "comic.jpg"),
    ]
    
    # Download content images
    print("Downloading content images...")
    for url, filename in content_images:
        save_path = os.path.join(samples_dir, filename)
        download_image(url, save_path)
    
    # Download style images
    print("\nDownloading style reference images...")
    for url, filename in style_images:
        save_path = os.path.join(samples_dir, filename)
        download_image(url, save_path)
    
    print("\nAll sample images downloaded to the 'samples' directory.")
    print("\nExample usage:")
    print("python style_transfer.py --content samples/landscape.jpg --style samples/vangogh.jpg --output output_vangogh.png")
    print("python style_transfer.py --content samples/portrait.jpg --style samples/watercolor.jpg --output output_watercolor.png --strength 0.65")
    print("python style_transfer.py --content samples/city.jpg --prompt \"cyberpunk style, neon colors, futuristic\" --output output_cyberpunk.png")


if __name__ == "__main__":
    main() 