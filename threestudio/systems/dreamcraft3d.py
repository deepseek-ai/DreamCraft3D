import os
import random
import shutil
from dataclasses import dataclass, field
import cv2
import clip
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.misc import get_rank, get_device, load_module_weights
from threestudio.utils.perceptual import PerceptualLoss


@threestudio.register("dreamcraft3d-system")
class ImageConditionDreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture'].
        # Note that in the paper we consolidate 'coarse' and 'geometry' into a single phase called 'geometry-sculpting'.
        stage: str = "coarse"
        freq: dict = field(default_factory=dict)
        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)
        use_mixed_camera_config: bool = False
        control_guidance_type: str = ""
        control_guidance: dict = field(default_factory=dict)
        control_prompt_processor_type: str = ""
        control_prompt_processor: dict = field(default_factory=dict)
        visualize_samples: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.guidance_3d_type != "":
            self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(
                self.cfg.guidance_3d
            )
        else:
            self.guidance_3d = None
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        p_config = {}
        self.perceptual_loss = threestudio.find("perceptual-loss")(p_config)

        if not (self.cfg.control_guidance_type == ""):
            self.control_guidance = threestudio.find(self.cfg.control_guidance_type)(self.cfg.control_guidance)
            self.control_prompt_processor = threestudio.find(self.cfg.control_prompt_processor_type)(
                self.cfg.control_prompt_processor
            )
            self.control_prompt_utils = self.control_prompt_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "texture":
            render_out = self.renderer(**batch, render_mask=True)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str, render_type="rgb"):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "guidance"
        """

        gt_mask = batch["mask"]
        gt_rgb = batch["rgb"]
        gt_depth = batch["ref_depth"]
        gt_normal = batch["ref_normal"]
        mvp_mtx_ref = batch["mvp_mtx"]
        c2w_ref = batch["c2w4x4"]

        if guidance == "guidance":
            batch = batch["random_camera"]

        # Support rendering visibility mask
        batch["mvp_mtx_ref"] = mvp_mtx_ref
        batch["c2w_ref"] = c2w_ref

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "guidance"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        prompt_utils = self.prompt_processor()

        if guidance == "ref":
            if render_type == "rgb":
                # color loss. Use l2 loss in coarse and geometry satge; use l1 loss in texture stage.
                if self.C(self.cfg.loss.lambda_rgb) > 0:
                    gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                        1 - gt_mask.float()
                    )
                    pred_rgb = out["comp_rgb"]
                    if self.cfg.stage in ["coarse", "geometry"]:
                        set_loss("rgb", F.mse_loss(gt_rgb, pred_rgb))
                    else:
                        if self.cfg.stage == "texture":
                            grow_mask = F.max_pool2d(1 - gt_mask.float().permute(0, 3, 1, 2), (9, 9), 1, 4)
                            grow_mask = (1 - grow_mask).permute(0, 2, 3, 1)
                            set_loss("rgb", F.l1_loss(gt_rgb*grow_mask, pred_rgb*grow_mask))
                        else:
                            set_loss("rgb", F.l1_loss(gt_rgb, pred_rgb))

                # mask loss
                if self.C(self.cfg.loss.lambda_mask) > 0:
                    set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

                # mask binary cross loss
                if self.C(self.cfg.loss.lambda_mask_binary) > 0:
                    set_loss("mask_binary", F.binary_cross_entropy(
                    out["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5),
                    batch["mask"].float(),))

                # depth loss
                if self.C(self.cfg.loss.lambda_depth) > 0:
                    valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                    valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                    with torch.no_grad():
                        A = torch.cat(
                            [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                        )  # [B, 2]
                        X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                        valid_gt_depth = A @ X  # [B, 1]
                    set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

                # relative depth loss
                if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                    valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                    valid_pred_depth = out["depth"][gt_mask]  # [B,]
                    set_loss(
                        "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                    )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * gt_normal[gt_mask.squeeze(-1)]
                )  # [B, 3]
                # FIXME: reverse x axis
                pred_normal = out["comp_normal_viewspace"]
                pred_normal[..., 0] = 1 - pred_normal[..., 0]
                valid_pred_normal = (
                    2 * pred_normal[gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )

        elif guidance == "guidance" and self.true_global_step > self.cfg.freq.no_diff_steps:
            if self.cfg.stage == "geometry" and render_type == "normal":
                guidance_inp = out["comp_normal"]
            else:
                guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp,
                prompt_utils,
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
                mask=out["mask"] if "mask" in out else None,
            )
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    set_loss(name.split("_")[-1], value)

            if self.guidance_3d is not None:

                # FIXME: use mixed camera config
                if not self.cfg.use_mixed_camera_config or get_rank() % 2 == 0:
                    guidance_3d_out = self.guidance_3d(
                        out["comp_rgb"],
                        **batch,
                        rgb_as_latents=False,
                        guidance_eval=guidance_eval,
                    )
                    for name, value in guidance_3d_out.items():
                        if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                            self.log(f"train/{name}_3d", value)
                        if name.startswith("loss_"):
                           set_loss("3d_"+name.split("_")[-1], value)
                    # set_loss("3d_sd", guidance_out["loss_sd"])

        # Regularization
        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )

            if "lambda_eikonal" in self.cfg.loss and self.C(self.cfg.loss.lambda_eikonal) > 0:
                if "sdf_grad" not in out:
                    raise ValueError(
                        "SDF grad is required for eikonal loss, no normal is found in the output."
                    )
                set_loss(
                    "eikonal", (
                        (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                    ).mean()
                )
            
            if "lambda_z_variance"in self.cfg.loss and self.C(self.cfg.loss.lambda_z_variance) > 0:
                # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
                # helps reduce floaters and produce solid geometry
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                set_loss("z_variance", loss_z_variance)

        elif self.cfg.stage == "geometry":
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())
        elif self.cfg.stage == "texture":
            if self.C(self.cfg.loss.lambda_reg) > 0 and guidance == "guidance" and self.true_global_step % 5 == 0:
            
                rgb = out["comp_rgb"]
                rgb = F.interpolate(rgb.permute(0, 3, 1, 2), (512, 512), mode='bilinear').permute(0, 2, 3, 1)
                control_prompt_utils = self.control_prompt_processor()
                with torch.no_grad():
                    control_dict = self.control_guidance(
                        rgb=rgb,
                        cond_rgb=rgb,
                        prompt_utils=control_prompt_utils,
                        mask=out["mask"] if "mask" in out else None,
                    )

                    edit_images = control_dict["edit_images"]
                    temp = (edit_images.detach().cpu()[0].numpy() * 255).astype(np.uint8)
                    cv2.imwrite(".threestudio_cache/control_debug.jpg", temp[:, :, ::-1])

                loss_reg = (rgb.shape[1] // 8) * (rgb.shape[2] // 8) * self.perceptual_loss(edit_images.permute(0, 3, 1, 2), rgb.permute(0, 3, 1, 2)).mean()
                set_loss("reg", loss_reg)
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        if self.cfg.freq.ref_or_guidance == "accumulate":
            do_ref = True
            do_guidance = True
        elif self.cfg.freq.ref_or_guidance == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_guidance = not do_ref
            if hasattr(self.guidance.cfg, "only_pretrain_step"):
                if (self.guidance.cfg.only_pretrain_step > 0) and (self.global_step % self.guidance.cfg.only_pretrain_step) < (self.guidance.cfg.only_pretrain_step // 5):
                    do_guidance = True
                    do_ref = False

        if self.cfg.stage == "geometry":
            render_type = "rgb" if self.true_global_step % self.cfg.freq.n_rgb == 0 else "normal"
        else:
            render_type = "rgb"

        total_loss = 0.0

        if do_guidance:
            out = self.training_substep(batch, batch_idx, guidance="guidance", render_type=render_type)
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref", render_type=render_type)
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)

        # sch = self.lr_schedulers()
        # sch.step()

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale", 
                        "img": out["depth"][0], 
                        "kwargs": {}
                    }
                ] 
                if "depth" in out
                else [] 
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            
            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.stage=="texture" and self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"

        try:
            self.save_img_sequence(
                filestem,
                filestem,
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="validation_epoch_end",
                step=self.true_global_step,
            )
            shutil.rmtree(
                os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
            )
        except:
            pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale", "img": out["depth"][0], "kwargs": {}
                        }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale", "img": out["opacity_vis"][0, :, :, 0], 
                        "kwargs": {"cmap": None, "data_range": (0, 1)}
                        }
                ]
                if "opacity_vis" in out
                else []
            )
            ,
            name="test_step",
            step=self.true_global_step,
        )

        # FIXME: save camera extrinsics
        c2w = batch["c2w"]
        save_path = os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test/{batch['index'][0]}.npy")
        np.save(save_path, c2w.detach().cpu().numpy()[0])

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        # print("on_before_opt enter")
        # for n, p in self.geometry.named_parameters():
        #     if p.grad is None:
        #         print(n)
        # print("on_before_opt exit")

        pass

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return 

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return 