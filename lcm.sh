#!/bin/bash

#hotshotxl

wget -c https://huggingface.co/hotshotco/SDXL-512/resolve/main/hsxl_base_1.0.safetensors -P ./models/vae/
wget -c https://huggingface.co/hotshotco/Hotshot-XL/resolve/main/hsxl_temporal_layers.safetensors -P ./models/vae/
wget -c https://huggingface.co/hotshotco/Hotshot-XL/resolve/main/hsxl_temporal_layers.f16.safetensors -P ./models/vae/
wget -c https://huggingface.co/hotshotco/SDXL-512/resolve/main/vae/diffusion_pytorch_model.safetensors -O ./models/vae/hotshotsdxl512_diffusion_pytorch_model.safetensors

