    #!/usr/bin/env python
"""
Instructions for running the Image Style Transfer app in Google Colab

== HOW TO USE THIS FILE ==

1. Create a new notebook in Google Colab
2. Copy and paste each code block below into separate cells in your notebook
3. Run the cells in sequence
4. Use the generated URL to access your app

== CODE BLOCKS FOR YOUR COLAB NOTEBOOK ==

# Block 1: Install required packages
!pip install torch torchvision diffusers==0.11.1 transformers accelerate numpy pillow matplotlib tqdm opencv-python ftfy scipy streamlit pyngrok

# Block 2: Create directory and navigate to it
!mkdir -p style_transfer_app
%cd style_transfer_app

# Block 3: Upload files (Run this cell and use the upload button that appears)
from google.colab import files
uploaded = files.upload()  # Upload style_transfer.py, app.py, and download_samples.py here

# Block 4: Create requirements.txt
%%writefile requirements.txt
torch>=1.13.0
torchvision>=0.14.0
diffusers>=0.11.1
transformers>=4.25.1
accelerate>=0.15.0
numpy>=1.23.5
Pillow>=9.3.0
matplotlib>=3.6.2
tqdm>=4.64.1
opencv-python>=4.6.0
ftfy>=6.1.1
scipy>=1.9.3
streamlit>=1.22.0

# Block 5: Download sample images
!python download_samples.py

# Block 6: Check GPU status
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU. This will be significantly slower.")
    print("Enable GPU: Runtime > Change runtime type > Hardware accelerator > GPU")

# Block 7: Fix matplotlib backend issue
import os
with open("style_transfer.py", "r") as file:
    content = file.read()
if "matplotlib.use('Agg')" not in content:
    modified_content = "import matplotlib\\nmatplotlib.use('Agg')\\n" + content
    with open("style_transfer.py", "w") as file:
        file.write(modified_content)
    print("Updated style_transfer.py to use non-interactive matplotlib backend")

# Block 8: Launch app with ngrok
from pyngrok import ngrok
import time

# Kill any existing Streamlit processes
!pkill -f streamlit || true
time.sleep(3)

# Start Streamlit in the background
!streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false &>/dev/null &
time.sleep(5)

# Create tunnel
ngrok_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
print(f"Access your app at: {ngrok_tunnel.public_url}")

# Block 9 (Alternative): Launch with localtunnel if ngrok doesn't work
# !npm install -g localtunnel
# !pkill -f streamlit || true
# !streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false &>/dev/null &
# !sleep 5 && lt --port 8501

# Block 10: Stop the app when done
# !pkill -f streamlit
"""

print("Please see the instructions and code blocks above for running in Google Colab")
print("Copy and paste each block into a separate cell in a Colab notebook")
