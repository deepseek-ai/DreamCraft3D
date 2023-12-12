from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm


SAVE_FOLDER = "./load/images_dreamfusion"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", default=0, type=int, help="# of GPU")
    parser.add_argument("--prompt",required=True, type=str)

    args = parser.parse_args()

    # stage 1
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_1.enable_model_cpu_offload()

    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    # stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_2.enable_model_cpu_offload()

    # stage 3
    # safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
    safety_modules = None
    stage_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16,
        local_files_only=True
    )
    stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_3.enable_model_cpu_offload()

    # # load prompt library
    # with open(os.path.join("load/prompt_library.json"), "r") as f:
    #     prompt_library = json.load(f)

    # n_prompts = len(prompt_library["dreamfusion"]) 
    # n_prompts_per_rank = int(np.ceil(n_prompts / 8))

    # for prompt in tqdm(prompt_library["dreamfusion"][args.rank * n_prompts_per_rank : (args.rank + 1) * n_prompts_per_rank]):
    
    prompt = args.prompt
    print("Prompt:", prompt)

    save_folder = os.path.join(SAVE_FOLDER, prompt)
    os.makedirs(save_folder, exist_ok=True)

    # if len(glob.glob(f"{save_folder}/*.png")) >= 30:
    #     continue

    # enhance prompt
    prompt = prompt + ", 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, hyperrealistic, intricate details, ultra-realistic, award-winning"

    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
    for _ in tqdm(range(30)):
        seed = np.random.randint(low=0, high=10000000, size=1)[0]
        generator = torch.manual_seed(seed)

        ### Stage 1
        image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        # pt_to_pil(image)[0].save("./if_stage_I.png")

        ### Stage 2
        image = stage_2(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
        ).images
        # pt_to_pil(image)[0].save("./if_stage_II.png")

        ### Stage 3
        image = stage_3(prompt=prompt, image=(image.float() * 0.5 + 0.5), generator=generator, noise_level=100).images
        image[0].save(f"{save_folder}/img_{seed:08d}.png")