import os
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision.utils import save_image
from subprocess import run, CalledProcessError
from threestudio.utils.config import load_config
import threestudio

# Constants
AZIMUTH_FACTOR = 360
IMAGE_SIZE = (512, 512)


def copy_file(source, destination):
    try:
        command = ['cp', source, destination]
        result = run(command, capture_output=True, text=True)
        result.check_returncode()
    except CalledProcessError as e:
        print(f'Error: {e.output}')


def prepare_images(cfg):
    rgb_list = sorted(glob.glob(os.path.join(cfg.data.render_image_path, "*.png")))
    rgb_list.sort(key=lambda file: int(os.path.splitext(os.path.basename(file))[0]))
    n_rgbs = len(rgb_list)
    n_samples = cfg.data.n_samples

    os.makedirs(cfg.data.save_path, exist_ok=True)

    copy_file(cfg.data.ref_image_path, f"{cfg.data.save_path}/ref_0.0.png")

    sampled_indices = np.linspace(0, len(rgb_list)-1, n_samples, dtype=int)
    rgb_samples = [rgb_list[index] for index in sampled_indices]

    return rgb_samples


def process_images(rgb_samples, cfg, guidance, prompt_utils):
    n_rgbs = 120
    for rgb_name in tqdm(rgb_samples):
        rgb_idx = int(os.path.basename(rgb_name).split(".")[0])
        rgb = cv2.imread(rgb_name)[:, :, :3][:, :, ::-1].copy() / 255.0
        H, W = rgb.shape[0:2]
        rgb_image, mask_image = rgb[:, :H], rgb[:, -H:, :1]
        rgb_image = cv2.resize(rgb_image, IMAGE_SIZE)
        rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)

        mask_image = cv2.resize(mask_image, IMAGE_SIZE).reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
        mask_image = torch.FloatTensor(mask_image).unsqueeze(0).to(guidance.device)

        temp = torch.zeros(1).to(guidance.device)
        azimuth = torch.tensor([rgb_idx/n_rgbs * AZIMUTH_FACTOR]).to(guidance.device)
        camera_distance = torch.tensor([cfg.data.default_camera_distance]).to(guidance.device)

        if cfg.data.view_dependent_noise:
            guidance.min_step_percent = 0. + (rgb_idx/n_rgbs) * (cfg.system.guidance.min_step_percent)
            guidance.max_step_percent = 0. + (rgb_idx/n_rgbs) * (cfg.system.guidance.max_step_percent)

        denoised_image = process_guidance(cfg, guidance, prompt_utils, rgb_image, azimuth, temp, camera_distance, mask_image)
        
        save_image(denoised_image.permute(0,3,1,2), f"{cfg.data.save_path}/img_{azimuth[0]}.png", normalize=True, value_range=(0, 1))

        copy_file(rgb_name.replace("png", "npy"), f"{cfg.data.save_path}/img_{azimuth[0]}.npy")

        if rgb_idx == 0:
            copy_file(rgb_name.replace("png", "npy"), f"{cfg.data.save_path}/ref_{azimuth[0]}.npy")       


def process_guidance(cfg, guidance, prompt_utils, rgb_image, azimuth, temp, camera_distance, mask_image):
    if cfg.data.azimuth_range[0] < azimuth < cfg.data.azimuth_range[1]:
        return guidance.sample_img2img(
            rgb_image, prompt_utils, temp, 
            azimuth, camera_distance, seed=0, mask=mask_image
        )["edit_image"]
    else:
        return rgb_image


def generate_mv_dataset(cfg):

    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(cfg.system.prompt_processor)
    prompt_utils = prompt_processor()

    guidance.update_step(epoch=0, global_step=0)
    rgb_samples = prepare_images(cfg)
    print(rgb_samples)
    process_images(rgb_samples, cfg, guidance, prompt_utils)

