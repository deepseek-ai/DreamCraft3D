import json
import os
from dataclasses import dataclass

import clip
import torch
import torch
import torch.nn as nn

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("clip-prompt-processor")
class ClipPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, device):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        clip_model, _ = clip.load(pretrained_model_name_or_path, jit=False)
        with torch.no_grad():
            tokens = clip.tokenize(
                prompts,
            ).to(device)
            text_embeddings = clip_model.encode_text(tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del clip_model
