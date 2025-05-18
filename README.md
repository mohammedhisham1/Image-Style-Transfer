# Image Style Transfer with Stable Diffusion

This project implements image style transfer using Stable Diffusion, allowing you to apply artistic styles from reference images to your content photos or use text prompts to guide the style transfer process.

## Features

- Transfer style from a reference image to a content image
- Use text prompts to guide style transfer
- Generate multiple style variations with different parameters
- Blend multiple reference styles
- Automatic style feature extraction from reference images
- Interactive visualization of results
- Streamlit web interface for easy use

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stable-diffusion-style-transfer.git
cd stable-diffusion-style-transfer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Download Sample Images

The project includes a utility script to download sample content and style images for testing:

```bash
python download_samples.py
```

This will create a `samples` directory with various content and style reference images.

## Usage

### Web Interface (Streamlit App)

The easiest way to use the style transfer tool is through the Streamlit web interface:

#### Windows:
```bash
run_app.bat
```

#### Linux/Mac:
```bash
chmod +x run_app.sh
./run_app.sh
```

Or directly with:
```bash
streamlit run app.py
```

This will launch a web application in your browser with the following features:
- Basic style transfer using a reference image
- Text-guided style transfer
- Generate multiple style variations
- Blend multiple styles
- Adjustable parameters (strength, guidance scale, steps)
- Download of generated images

### Quick Example

Run the example script to see different style transfer modes in action:

```bash
# Download sample images and run basic style transfer
python example.py --download --mode basic

# Try text-guided style transfer
python example.py --mode text

# Generate multiple variations
python example.py --mode variations

# Blend multiple styles
python example.py --mode blend
```

### Basic Style Transfer

Transfer the style from a reference image to a content image:

```bash
python style_transfer.py --content samples/landscape.jpg --style samples/vangogh.jpg --output output.png
```

### Text-Guided Style Transfer

Use a text prompt to guide the style transfer without a reference image:

```bash
python style_transfer.py --content samples/portrait.jpg --prompt "watercolor painting, soft colors, artistic, elegant" --output output_watercolor.png
```

### Generate Multiple Variations

Create multiple style variations with slightly different parameters:

```bash
python style_transfer.py --content samples/city.jpg --style samples/comic.jpg --variations 3 --output variations/output.png
```

### Additional Options

- `--strength`: Control the style transfer strength (0.0-1.0, default: 0.75)
- `--steps`: Number of inference steps (default: 50)
- `--guidance`: Guidance scale (default: 7.5)
- `--model`: Change the Stable Diffusion model (default: runwayml/stable-diffusion-v1-5)
- `--device`: Specify the device to use (cuda or cpu)

## Advanced Usage

The project can also be used as a module in your own Python code:

```python
from PIL import Image
from style_transfer import StyleTransfer

# Initialize the style transfer class
transfer = StyleTransfer()

# Load images
content_image = Image.open("content.jpg")
style_image = Image.open("style.jpg")

# Transfer style
result = transfer.transfer_style(
    content_image, 
    style_image,
    strength=0.75,
    guidance_scale=7.5,
    num_inference_steps=50
)

# Save result
result.save("output.png")

# Visualize comparison
transfer.visualize_comparison(content_image, style_image, result)
```

## How It Works

The style transfer process works by:

1. Analyzing the reference style image to extract color, texture, and pattern information
2. Converting these visual features into a text prompt that captures the essence of the style
3. Using Stable Diffusion's img2img pipeline to apply the style to the content image
4. The strength parameter controls how much of the original content is preserved

## Requirements

- CUDA-capable GPU with at least 4GB VRAM (for faster processing)
- Python 3.8 or higher
- See requirements.txt for Python package dependencies

## License

This project is available under the MIT License. See the LICENSE file for details. 