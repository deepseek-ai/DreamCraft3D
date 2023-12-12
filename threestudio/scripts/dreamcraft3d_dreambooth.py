import argparse
import os
from subprocess import run, CalledProcessError

import cv2
import glob
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from threestudio.scripts.generate_mv_datasets import generate_mv_dataset
from threestudio.utils.config import load_config
import threestudio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--action", default="both", help="action to perform", choices=["gen_data", "dreambooth", "both""])
    args, extras = parser.parse_known_args()
    return args, extras


def main(args, extras):
    cfg = load_config(args.config, cli_args=extras, n_gpus=1)

    if args.action == "gen_data" or args.action == "both":
        # Generate multi-view dataset
        generate_mv_dataset(cfg)

    if args.action == "dreambooth" or args.action == "both":
        # Run DreamBooth.
        command = f'accelerate launch threestudio/scripts/train_dreambooth.py \
                    --pretrained_model_name_or_path="{cfg.custom_import.dreambooth.model_name}" \
                    --instance_data_dir="{cfg.custom_import.dreambooth.instance_dir}" \
                    --output_dir="{cfg.custom_import.dreambooth.output_dir}"\
                    --instance_prompt="{cfg.custom_import.dreambooth.prompt_dreambooth}" \
                    --resolution=512 \
                    --train_batch_size=2 \
                    --gradient_accumulation_steps=1 \
                    --learning_rate=1e-6 \
                    --lr_scheduler="constant" \
                    --lr_warmup_steps=0 \
                    --max_train_steps=1000'

        os.system(command)


if __name__ == "__main__":
    args, extras = parse_args()
    main(args, extras)
