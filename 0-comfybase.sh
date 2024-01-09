#!/bin/bash

# stable pytorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Clone the ComfyUI repository from GitHub
git clone https://github.com/comfyanonymous/ComfyUI

# Change into the cloned directory
cd ComfyUI

# Install the required packages using pip
pip install -r requirements.txt
