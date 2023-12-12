from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

# model_id = "load/checkpoints/sd_21_base_mushroom_vd_prompt"
# model_id = "load/checkpoints/sd_base_mushroom"
model_id = ".cache/checkpoints/sd_21_base_rabbit"
# scheduler = DDIMScheduler()
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
guidance_scale = 7.5

prompt = "a sks rabbit, front view"
image = pipe(prompt, num_inference_steps=50, guidance_scale=guidance_scale).images[0]

image.save("debug.png")


# import os
# import cv2
# import glob
# import torch
# import argparse
# import numpy as np
# from tqdm import tqdm
# import pytorch_lightning as pl
# from torchvision.utils import save_image

# import threestudio
# from threestudio.utils.config import load_config


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True, help="path to config file")
#     parser.add_argument("--view_dependent_noise", action="store_true", help="use view depdendent noise strength")

#     args, extras = parser.parse_known_args()

#     cfg = load_config(args.config, cli_args=extras, n_gpus=1)
#     guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
#     prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(cfg.system.prompt_processor)
#     prompt_utils = prompt_processor()

#     guidance.update_step(epoch=0, global_step=0)
#     elevation, azimuth = torch.zeros(1).cuda(), torch.zeros(1).cuda()
#     camera_distances = torch.tensor([3.0]).cuda()
#     c2w = torch.zeros(4,4).cuda()
#     a = guidance.sample(prompt_utils, elevation, azimuth, camera_distances) # sample_lora
#     from torchvision.utils import save_image
#     save_image(a.permute(0,3,1,2), "debug.png", normalize=True, value_range=(0,1))



# python threestudio/scripts/test_dreambooth.py --config configs/experimental/stablediffusion.yaml system.prompt_processor.prompt="a sks mushroom growing on a log" \
#     system.guidance.pretrained_model_name_or_path_lora="load/checkpoints/sd_21_base_mushroom_camera_condition"