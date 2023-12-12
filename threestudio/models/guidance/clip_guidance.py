from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import clip

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.typing import *


@threestudio.register("clip-guidance")
class CLIPGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "ViT-B/16"
        view_dependent_prompting: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading CLIP ...")
        self.clip_model, self.clip_preprocess = clip.load(
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            jit=False,
            download_root=self.cfg.cache_dir
        )

        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        threestudio.info(f"Loaded CLIP!")

    @torch.cuda.amp.autocast(enabled=False)
    def get_embedding(self, input_value, is_text=True):
        if is_text:
            value = clip.tokenize(input_value).to(self.device)
            z = self.clip_model.encode_text(value)
        else:
            input_value = self.aug(input_value)
            z = self.clip_model.encode_image(input_value)

        return z / z.norm(dim=-1, keepdim=True)

    def get_loss(self, image_z, clip_z, loss_type='similarity_score', use_mean=True):
        if loss_type == 'similarity_score':
            loss = -((image_z * clip_z).sum(-1))
        elif loss_type == 'spherical_dist':
            image_z, clip_z = F.normalize(image_z, dim=-1), F.normalize(clip_z, dim=-1)
            loss = ((image_z - clip_z).norm(dim=-1).div(2).arcsin().pow(2).mul(2))
        else:
            raise NotImplementedError

        return loss.mean() if use_mean else loss

    def __call__(
        self,
        pred_rgb: Float[Tensor, "B H W C"],
        gt_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        embedding_type: str = 'both',
        loss_type: Optional[str] = 'similarity_score',
        **kwargs,
    ):
        clip_text_loss, clip_img_loss = 0, 0

        if embedding_type in ('both', 'text'):
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            ).chunk(2)[0]
            clip_text_loss = self.get_loss(self.get_embedding(pred_rgb, is_text=False), text_embeddings, loss_type=loss_type)

        if embedding_type in ('both', 'img'):
            clip_img_loss = self.get_loss(self.get_embedding(pred_rgb, is_text=False), self.get_embedding(gt_rgb, is_text=False), loss_type=loss_type)

        return clip_text_loss + clip_img_loss
