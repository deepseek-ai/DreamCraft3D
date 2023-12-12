import os
import argparse
from PIL import Image
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionUpscalePipeline


def load_model(superres):
    mv_model = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16, cache_dir="load/checkpoints/huggingface/hub", local_files_only=True,
    )
    mv_model.scheduler = EulerAncestralDiscreteScheduler.from_config(
        mv_model.scheduler.config, timestep_spacing='trailing', cache_dir="load/checkpoints/huggingface/hub", local_files_only=True,
    )

    if superres:
        superres_model = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", revision="fp16",
            torch_dtype=torch.float16, cache_dir="load/checkpoints/huggingface/hub", local_files_only=True,
        )
    else:
        superres_model = None

    return mv_model, superres_model

    
def superres_4x(image, model, prompt):
    low_res_img = image.resize((256, 256))
    model.to('cuda:1')
    result = model(prompt=prompt, image=low_res_img).images[0]
    return result


def img_to_mv(image_path, model):
    cond = Image.open(image_path)
    model.to('cuda:1')
    result = model(cond, num_inference_steps=75).images[0]
    return result


def crop_save_image_to_2x3_grid(image, args, model):
    save_path = args.save_path
    width, height = image.size
    grid_width = width//2
    grid_height = height//3

    images = []
    for i in range(3):
        for j in range(2):
            left = j * grid_width
            upper = i * grid_height
            right = (j+1) * grid_width
            lower = (i+1) * grid_height
            
            cropped_image = image.crop((left, upper, right, lower))
            if args.superres:
                cropped_image = superres_4x(cropped_image, model, args.prompt)
            images.append(cropped_image)

    for idx, img in enumerate(images):
        img.save(os.path.join(save_path, f'cropped_{idx}.jpg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--save_path', type=str, help="path to save output images")
    parser.add_argument('--prompt', type=str, help="prompt to use for superres")
    parser.add_argument('--superres', action='store_true', help="whether to use superres")
    args = parser.parse_args()

    print(args.superres)

    os.makedirs(args.save_path, exist_ok=True)
    os.system(f"cp '{args.image_path}' '{args.save_path}'")

    mv_model, superres_model = load_model(args.superres)
    images = img_to_mv(args.image_path, mv_model)
    crop_save_image_to_2x3_grid(images, args, superres_model)


# Example usage:
# python threestudio/scripts/img_to_mv.py --image_path 'mushroom.png' --save_path '.cache/temp' --prompt 'a photo of mushroom' --superres