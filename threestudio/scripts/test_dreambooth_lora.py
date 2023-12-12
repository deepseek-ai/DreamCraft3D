import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


# model_base = "stabilityai/stable-diffusion-2-1-base"

# pipe = DiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, cache_dir=CACHE_DIR, local_files_only=True)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, cache_dir=CACHE_DIR, local_files_only=True)
# lora_model_path = "load/checkpoints/sd_21_base_bear_dreambooth_lora"
# pipe.unet.load_attn_procs(lora_model_path)

# pipe.to("cuda")


# image = pipe("A picture of a sks bear in the sky", num_inference_steps=50, guidance_scale=7.5).images[0]
# image.save("bear_dreambooth_lora.png")


pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", local_files_only=True, safety_checker=None)
pipe.load_lora_weights("if_dreambooth_mushroom")
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
pipe.to("cuda:7")

image = pipe("A photo of a sks mushroom, front view", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("mushroom_dreambooth_lora.png")