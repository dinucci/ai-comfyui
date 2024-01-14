#!/bin/bash

# Video Diffusion
#wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd_image_decoder.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/svd.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors -P ./models/checkpoints/

# Checkpoints

### SDXL Turbo
wget -c https://civitai.com/api/download/models/272378 -O ./models/checkpoints/sdxl_turbo-realvisxl-v30-turbo.safetensors
wget -c https://civitai.com/api/download/models/273102 -O ./models/checkpoints/sdxl_turbo-turbovisionxl-v431-bakedVAE.safetensors

### SDXL
### I recommend these workflow examples: https://comfyanonymous.github.io/ComfyUI_examples/sdxl/
wget -c https://civitai.com/api/download/models/289071 -O ./models/checkpoints/sdxl-yamers-realistic-nsfw-and-sfw.safetensors
wget -c https://civitai.com/api/download/models/288982 -O ./models/checkpoints/sdxl-juggernaut-xl.safetensors
wget -c https://civitai.com/api/download/models/275491 -O ./models/checkpoints/sdxl-jnewrealityxl-photographic-all-in-one.safetensors
wget -c https://civitai.com/api/download/models/293413 -O ./models/checkpoints/sdxl-copax-timelessxl-v9-sdxl10.safetensors
wget -c https://civitai.com/api/download/models/275352 -O ./models/checkpoints/sdxl_pixelwave.safetensors
wget -c https://civitai.com/api/download/models/293240 -O ./models/checkpoints/sdxl_realism-engine-sdxl.safetensors
wget -c https://civitai.com/api/download/models/286100 -O ./models/checkpoints/sdxl_rmohawk.safetensors

#wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/latent-consistency/lcm-sdxl/resolve/main/diffusion_pytorch_model.fp16.safetensors -O ./models/loras/lcm-sdxl_diffusion_pytorch_model-fp16.safetensors
wget -c https://civitai.com/api/download/models/240840?type=Model&format=SafeTensor&size=full&fp=fp16 -O ./models/checkpoints/juggernautxlv7.safetensors
wget -c https://huggingface.co/ckpt/cardos-animated/resolve/main/cardosAnimated_v20.safetensors -P ./models/checkpoints/


# SDXL ReVision
wget -c https://huggingface.co/comfyanonymous/clip_vision_g/resolve/main/clip_vision_g.safetensors -P ./models/clip_vision/

# SD1.5
#wget -c https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -P ./models/checkpoints/
#wget -c https://civitai.com/api/download/models/272376 -O ./models/checkpoints/sd15_picxreal.safetensors
#wget -c https://civitai.com/api/download/models/286170 -O ./models/checkpoints/sd15_reallife.safetensors
#

# SD2
#wget -c https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors -P ./models/checkpoints/

# Some SD1.5 anime style
#wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix2/AbyssOrangeMix2_hard.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix3/AOM3A1_orangemixs.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix3/AOM3A3_orangemixs.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/anything-v3-fp16-pruned.safetensors -P ./models/checkpoints/

# Waifu Diffusion 1.5 (anime style SD2.x 768-v)
#wget -c https://huggingface.co/waifu-diffusion/wd-1-5-beta3/resolve/main/wd-illusion-fp16.safetensors -P ./models/checkpoints/


# unCLIP models
wget -c https://huggingface.co/comfyanonymous/illuminatiDiffusionV1_v11_unCLIP/resolve/main/illuminatiDiffusionV1_v11-unclip-h-fp16.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-aesthetic-unclip-h-fp16.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/resolve/main/sd21-unclip-h.ckpt -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/resolve/main/sd21-unclip-l.ckpt -P ./models/checkpoints/

# unCLIP SD 1.5
wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-aesthetic-unclip-h-fp16.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-aesthetic-unclip-h-fp32.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-aesthetic-unclip-l-fp16.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-aesthetic-unclip-l-fp32.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-unclip-h-fp16.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-unclip-h-fp32.safetensors -P ./models/checkpoints/

# VAE
wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -P ./models/vae/
wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt -P ./models/vae/
wget -c https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt -P ./models/vae/
wget -c https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors -P ./models/vae/


# Loras
wget -c https://civitai.com/api/download/models/10350 -O ./models/loras/theovercomer8sContrastFix_sd21768.safetensors #theovercomer8sContrastFix SD2.x 768-v
wget -c https://civitai.com/api/download/models/10638 -O ./models/loras/theovercomer8sContrastFix_sd15.safetensors #theovercomer8sContrastFix SD1.x
wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors -P ./models/loras/ #SDXL offset noise lora


# T2I-Adapter
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth -P ./models/controlnet/
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_seg_sd14v1.pth -P ./models/controlnet/
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth -P ./models/controlnet/
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_keypose_sd14v1.pth -P ./models/controlnet/
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_openpose_sd14v1.pth -P ./models/controlnet/
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_color_sd14v1.pth -P ./models/controlnet/
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_canny_sd14v1.pth -P ./models/controlnet/

# T2I Styles Model
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_style_sd14v1.pth -P ./models/style_models/

# CLIPVision model (needed for styles model)
wget -c https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin -O ./models/clip_vision/clip_vit14.bin


# ControlNet
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_seg_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors -P ./models/controlnet/

# ControlNet SDXL
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors -P ./models/controlnet/

# Controlnet Preprocessor nodes by Fannovel16
cd custom_nodes && git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors; cd comfy_controlnet_preprocessors && python install.py


# GLIGEN
#wget -c https://huggingface.co/comfyanonymous/GLIGEN_pruned_safetensors/resolve/main/gligen_sd14_textbox_pruned_fp16.safetensors -P ./models/gligen/


# ESRGAN upscale model
wget -c https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./models/upscale_models/
wget -c https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth -P ./models/upscale_models/
wget -c https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth -P ./models/upscale_models/

# Upscaler

wget -c https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth -P ./models/upscale_models/
wget -c https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth -P ./models/upscale_models/
