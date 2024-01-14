#!/bin/bash

# AnimateDiff
wget -c https://huggingface.co/manshoety/AD_Stabilized_Motion/resolve/main/mm-Stabilized_mid.pth -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
wget -c https://huggingface.co/manshoety/AD_Stabilized_Motion/resolve/main/mm-Stabilized_high.pth -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
wget -c https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd15_v3.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
wget -c https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
wget -c https://huggingface.co/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
