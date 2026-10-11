from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_backbone_with_fpn(
    cfg: dict,
    extra_blocks: nn.Module | None = None,
    returned_layers: list[int] | None = None,
) -> nn.Module:
    backbone_cfg = cfg["backbone"]
    name = backbone_cfg["name"]
    if name != "resnet50":
        raise ValueError(f"Unsupported backbone: {name!r}. Choose from ['resnet50']")
    pretrained = backbone_cfg.get("pretrained")
    return resnet_fpn_backbone(
        backbone_name=name,
        weights=None if pretrained is None else getattr(ResNet50_Weights, pretrained),
        trainable_layers=backbone_cfg.get("trainable_layers", 3),
        returned_layers=returned_layers,
        extra_blocks=extra_blocks,
    )


class BaseDetectionWrapper(nn.Module):
    model: nn.Module

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        return self.model(images, targets)
