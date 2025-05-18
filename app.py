#!/usr/bin/env python
# Streamlit app for Image Style Transfer

import os
import time
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

from style_transfer import StyleTransfer

# Page configuration
st.set_page_config(
    page_title="Image Style Transfer with Stable Diffusion",
    page_icon="🎨",
    layout="wide",
)

# Function to load and display images
@st.cache_data
def load_image(image_file):
    if image_file is not None:
        img = Image.open(image_file)
        return img
    return None

# Function to convert PIL image to bytes for download
def image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# Main function
def main():
    # Title and description
    st.title("🎨 Image Style Transfer with Stable Diffusion")
    st.markdown(
        """
        Transfer artistic styles to your images using Stable Diffusion. Upload your content image 
        and choose one of the available methods to stylize it.
        """
    )
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Model selector in sidebar
    model_id = st.sidebar.selectbox(
        "Select Stable Diffusion model",
        ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1-base"],
        index=0,
    )
    
    # Device selector in sidebar
    device = st.sidebar.radio(
        "Select device",
        ["CUDA (GPU)", "CPU"],
        index=0 if torch.cuda.is_available() else 1,
    )
    device = "cuda" if device == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"
    
    # Style transfer mode selector
    transfer_mode = st.sidebar.radio(
        "Transfer Mode",
        ["Basic Style Transfer", "Text-Guided Style", "Style Variations", "Blend Styles"],
        index=0,
    )
    
    # Initialize Style Transfer
    @st.cache_resource(show_spinner="Loading Stable Diffusion model...")
    def get_transfer_model(model_id, device):
        return StyleTransfer(model_id=model_id, device=device)
    
    transfer = get_transfer_model(model_id, device)
    
    # Display mode-specific UI
    if transfer_mode == "Basic Style Transfer":
        basic_style_transfer(transfer)
    elif transfer_mode == "Text-Guided Style":
        text_guided_transfer(transfer)
    elif transfer_mode == "Style Variations":
        style_variations(transfer)
    else:  # Blend Styles
        blend_styles(transfer)
    


def basic_style_transfer(transfer):
    st.header("Basic Style Transfer")
    st.markdown(
        """
        Upload a content image and a style reference image to transfer the style from the reference to your content.
        """
    )
    
    # Create two columns for content and style images
    col1, col2 = st.columns(2)
    
    # Content image upload
    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader("Upload your content image", type=["jpg", "jpeg", "png"], key="content_basic")
        content_image = None
        if content_file is not None:
            content_image = load_image(content_file)
            st.image(content_image, caption="Content Image", use_column_width=True)
    
    # Style image upload
    with col2:
        st.subheader("Style Reference")
        style_file = st.file_uploader("Upload your style reference", type=["jpg", "jpeg", "png"], key="style_basic")
        style_image = None
        if style_file is not None:
            style_image = load_image(style_file)
            st.image(style_image, caption="Style Reference", use_column_width=True)
    
    # Parameters
    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        strength = st.slider("Style Strength", 0.1, 0.9, 0.75, 0.05, 
                           help="How much to transform the image (higher means more stylized)")
    
    with col2:
        guidance_scale = st.slider("Guidance Scale", 5.0, 10.0, 7.5, 0.5,
                                help="How closely to follow the prompt (higher means more precise)")
    
    inference_steps = st.slider("Inference Steps", 20, 100, 50, 10,
                              help="More steps = better quality but slower")
    
    # Process button
    if st.button("Generate Stylized Image", type="primary", disabled=(content_image is None or style_image is None)):
        if content_image is not None and style_image is not None:
            with st.spinner("Generating stylized image... This may take a while."):
                try:
                    # Process the image
                    start_time = time.time()
                    result = transfer.transfer_style(
                        content_image,
                        style_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=inference_steps
                    )
                    end_time = time.time()
                    
                    # Display the result
                    st.subheader("Result")
                    st.image(result, caption="Stylized Image", use_column_width=True)
                    
                    # Show processing time
                    st.success(f"Processing completed in {end_time - start_time:.2f} seconds")
                    
                    # Download button
                    st.download_button(
                        label="Download Stylized Image",
                        data=image_to_bytes(result),
                        file_name="stylized_image.png",
                        mime="image/png",
                    )
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
        else:
            st.warning("Please upload both content and style images.")


def text_guided_transfer(transfer):
    st.header("Text-Guided Style Transfer")
    st.markdown(
        """
        Upload a content image and provide a text prompt describing the style you want to apply.
        """
    )
    
    # Content image upload
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload your content image", type=["jpg", "jpeg", "png"], key="content_text")
    content_image = None
    if content_file is not None:
        content_image = load_image(content_file)
        st.image(content_image, caption="Content Image", width=400)
    
    # Text prompt
    st.subheader("Style Description")
    text_prompt = st.text_area(
        "Describe the style you want to apply",
        "watercolor painting, soft colors, artistic, elegant, detailed brushstrokes",
        help="Be descriptive about the style, colors, textures, etc."
    )
    
    # Parameters
    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        strength = st.slider("Style Strength", 0.1, 0.9, 0.7, 0.05,
                           help="How much to transform the image (higher means more stylized)",
                           key="text_strength")
    
    with col2:
        guidance_scale = st.slider("Guidance Scale", 5.0, 12.0, 8.0, 0.5,
                                help="How closely to follow the prompt (higher means more precise)",
                                key="text_guidance")
    
    inference_steps = st.slider("Inference Steps", 20, 100, 50, 10,
                              help="More steps = better quality but slower",
                              key="text_steps")
    
    # Process button
    if st.button("Generate Stylized Image", type="primary", disabled=content_image is None, key="text_generate"):
        if content_image is not None:
            with st.spinner("Generating stylized image... This may take a while."):
                try:
                    # Process the image
                    start_time = time.time()
                    result = transfer.transfer_style_with_prompt(
                        content_image,
                        text_prompt,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=inference_steps
                    )
                    end_time = time.time()
                    
                    # Display the result
                    st.subheader("Result")
                    
                    # Show side by side comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(content_image, caption="Original", use_column_width=True)
                    with col2:
                        st.image(result, caption="Stylized", use_column_width=True)
                    
                    # Show processing time
                    st.success(f"Processing completed in {end_time - start_time:.2f} seconds")
                    
                    # Download button
                    st.download_button(
                        label="Download Stylized Image",
                        data=image_to_bytes(result),
                        file_name="text_stylized_image.png",
                        mime="image/png",
                        key="text_download"
                    )
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
        else:
            st.warning("Please upload a content image.")


def style_variations(transfer):
    st.header("Style Variations")
    st.markdown(
        """
        Upload a content image and a style reference to generate multiple variations with different parameters.
        """
    )
    
    # Create two columns for content and style images
    col1, col2 = st.columns(2)
    
    # Content image upload
    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader("Upload your content image", type=["jpg", "jpeg", "png"], key="content_var")
        content_image = None
        if content_file is not None:
            content_image = load_image(content_file)
            st.image(content_image, caption="Content Image", use_column_width=True)
    
    # Style image upload
    with col2:
        st.subheader("Style Reference")
        style_file = st.file_uploader("Upload your style reference", type=["jpg", "jpeg", "png"], key="style_var")
        style_image = None
        if style_file is not None:
            style_image = load_image(style_file)
            st.image(style_image, caption="Style Reference", use_column_width=True)
    
    # Parameters
    st.subheader("Variation Parameters")
    num_variations = st.slider("Number of Variations", 2, 5, 3, 1,
                             help="How many different variations to generate")
    
    col1, col2 = st.columns(2)
    with col1:
        min_strength = st.slider("Min Strength", 0.1, 0.7, 0.6, 0.05,
                              help="Minimum style strength for variations")
    
    with col2:
        max_strength = st.slider("Max Strength", min_strength + 0.1, 0.9, 0.8, 0.05,
                              help="Maximum style strength for variations")
    
    # Process button
    if st.button("Generate Variations", type="primary", disabled=(content_image is None or style_image is None),
              key="var_generate"):
        if content_image is not None and style_image is not None:
            with st.spinner(f"Generating {num_variations} style variations... This may take a while."):
                try:
                    # Process the image
                    start_time = time.time()
                    variations = transfer.create_style_variations(
                        content_image,
                        style_image,
                        variations=num_variations,
                        strength_range=(min_strength, max_strength)
                    )
                    end_time = time.time()
                    
                    # Display the results
                    st.subheader("Style Variations")
                    
                    # Show all variations in a grid
                    cols = st.columns(num_variations)
                    for i, (col, img) in enumerate(zip(cols, variations)):
                        with col:
                            st.image(img, caption=f"Variation {i+1}", use_column_width=True)
                            st.download_button(
                                label=f"Download #{i+1}",
                                data=image_to_bytes(img),
                                file_name=f"variation_{i+1}.png",
                                mime="image/png",
                                key=f"var_download_{i}"
                            )
                    
                    # Show processing time
                    st.success(f"Processing completed in {end_time - start_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
        else:
            st.warning("Please upload both content and style images.")


def blend_styles(transfer):
    st.header("Blend Multiple Styles")
    st.markdown(
        """
        Upload a content image and two style references to blend them together.
        """
    )
    
    # Content image upload
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload your content image", type=["jpg", "jpeg", "png"], key="content_blend")
    content_image = None
    if content_file is not None:
        content_image = load_image(content_file)
        st.image(content_image, caption="Content Image", width=400)
    
    # Style images upload
    st.subheader("Style References")
    col1, col2 = st.columns(2)
    
    with col1:
        style1_file = st.file_uploader("Upload first style reference", type=["jpg", "jpeg", "png"], key="style1_blend")
        style1_image = None
        if style1_file is not None:
            style1_image = load_image(style1_file)
            st.image(style1_image, caption="Style 1", use_column_width=True)
    
    with col2:
        style2_file = st.file_uploader("Upload second style reference", type=["jpg", "jpeg", "png"], key="style2_blend")
        style2_image = None
        if style2_file is not None:
            style2_image = load_image(style2_file)
            st.image(style2_image, caption="Style 2", use_column_width=True)
    
    # Blending weight
    st.subheader("Blending Parameters")
    style1_weight = st.slider("Style 1 Weight", 0.0, 1.0, 0.7, 0.1,
                           help="How much to weight the first style (Style 2 will be 1 - this value)")
    
    # Calculate style2 weight
    style2_weight = round(1.0 - style1_weight, 1)
    st.caption(f"Style 2 Weight: {style2_weight}")
    
    # Additional parameters
    col1, col2 = st.columns(2)
    with col1:
        strength = st.slider("Overall Strength", 0.1, 0.9, 0.75, 0.05,
                          help="How much to transform the image (higher means more stylized)",
                          key="blend_strength")
    
    with col2:
        guidance_scale = st.slider("Guidance Scale", 5.0, 10.0, 7.5, 0.5,
                               help="How closely to follow the prompt (higher means more precise)",
                               key="blend_guidance")
    
    inference_steps = st.slider("Inference Steps", 20, 100, 50, 10,
                             help="More steps = better quality but slower",
                             key="blend_steps")
    
    # Process button
    if st.button("Blend Styles", type="primary", 
              disabled=(content_image is None or style1_image is None or style2_image is None),
              key="blend_generate"):
        if content_image is not None and style1_image is not None and style2_image is not None:
            with st.spinner("Blending styles... This may take a while."):
                try:
                    # Process the image
                    start_time = time.time()
                    result = transfer.blend_styles(
                        content_image,
                        [style1_image, style2_image],
                        weights=[style1_weight, style2_weight],
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=inference_steps
                    )
                    end_time = time.time()
                    
                    # Display the result
                    st.subheader("Blended Result")
                    st.image(result, caption="Blended Stylized Image", use_column_width=False, width=600)
                    
                    # Show processing time
                    st.success(f"Processing completed in {end_time - start_time:.2f} seconds")
                    
                    # Download button
                    st.download_button(
                        label="Download Blended Image",
                        data=image_to_bytes(result),
                        file_name="blended_stylized_image.png",
                        mime="image/png",
                        key="blend_download"
                    )
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
        else:
            st.warning("Please upload the content image and both style references.")


if __name__ == "__main__":
    main() 