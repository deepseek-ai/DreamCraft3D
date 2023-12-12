from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        render_mask: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        if render_mask:
            # get front-view visibility mask
            with torch.no_grad():
                mvp_mtx_ref = kwargs["mvp_mtx_ref"] # FIXME
                v_pos_clip_front: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                    mesh.v_pos, mvp_mtx_ref
                )
                rast_front, _ = self.ctx.rasterize(v_pos_clip_front, mesh.t_pos_idx, (height, width))
                mask_front = rast_front[..., 3:]
                mask_front = mask_front[mask_front > 0] - 1.
                faces_vis = mesh.t_pos_idx[mask_front.long()]

                mesh._v_rgb = torch.zeros(mesh.v_pos.shape[0], 1).to(mesh.v_pos)
                mesh._v_rgb[faces_vis[:,0]] = 1.
                mesh._v_rgb[faces_vis[:,1]] = 1.
                mesh._v_rgb[faces_vis[:,2]] = 1.
                mask_vis, _ = self.ctx.interpolate_one(mesh._v_rgb, rast, mesh.t_pos_idx)
                mask_vis = mask_vis > 0.
                # from torchvision.utils import save_image
                # save_image(mask_vis.permute(0,3,1,2).float(), "debug.png")
                out.update({"mask": 1.0 - mask_vis.float()})
                
                # FIXME: paste texture back to mesh
                # import cv2
                # import imageio
                # import numpy as np

                # gt_rgb = imageio.imread("load/images/tiger_nurse_rgba.png")/255.
                # gt_rgb = cv2.resize(gt_rgb[:,:,:3],(512, 512))
                # gt_rgb = torch.Tensor(gt_rgb[None,...]).permute(0,3,1,2).to(v_pos_clip_front)

                # # align to up-z and front-x
                # dir2vec = {
                #     "+x": np.array([1, 0, 0]),
                #     "+y": np.array([0, 1, 0]),
                #     "+z": np.array([0, 0, 1]),
                #     "-x": np.array([-1, 0, 0]),
                #     "-y": np.array([0, -1, 0]),
                #     "-z": np.array([0, 0, -1]),
                # }
                # z_, x_ = (
                #     dir2vec["-y"],
                #     dir2vec["-z"],
                # )

                # y_ = np.cross(z_, x_)
                # std2mesh = np.stack([x_, y_, z_], axis=0).T
                # v_pos_ = (torch.mm(torch.tensor(std2mesh).to(mesh.v_pos), mesh.v_pos.T).T) * 2
                # print(v_pos_.min(), v_pos_.max())

                # mesh._v_rgb=F.grid_sample(gt_rgb, v_pos_[None, None][..., :2], mode="nearest").permute(3,1,0,2).squeeze(-1).squeeze(-1).contiguous()
                # rgb_vis, _ = self.ctx.interpolate_one(mesh._v_rgb, rast, mesh.t_pos_idx)
                # rgb_vis_aa = self.ctx.antialias(
                #     rgb_vis, rast, v_pos_clip, mesh.t_pos_idx
                # )
                # from torchvision.utils import save_image
                # save_image(rgb_vis_aa.permute(0,3,1,2), "debug.png")
        

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # Compute normal in view space.
        # TODO: make is clear whether to compute this.
        w2c = kwargs["c2w"][:, :3, :3].inverse()
        gb_normal_viewspace = torch.einsum("bij,bhwj->bhwi", w2c, gb_normal)
        gb_normal_viewspace = F.normalize(gb_normal_viewspace, dim=-1)
        bg_normal = torch.zeros_like(gb_normal_viewspace)
        bg_normal[..., 2] = 1
        gb_normal_viewspace_aa = torch.lerp(
            (bg_normal + 1.0) / 2.0,
            (gb_normal_viewspace + 1.0) / 2.0,
            mask.float(),
        ).contiguous()
        gb_normal_viewspace_aa = self.ctx.antialias(
            gb_normal_viewspace_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"comp_normal_viewspace": gb_normal_viewspace_aa})

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            rgb_fg = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out
            )
            gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg})

        return out