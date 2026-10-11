from __future__ import annotations

from torchvision.models.detection import FCOS
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from .base import BaseDetectionWrapper, build_backbone_with_fpn


class FCOSWrapper(BaseDetectionWrapper):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        backbone = build_backbone_with_fpn(
            cfg,
            extra_blocks=LastLevelP6P7(256, 256),
            returned_layers=[2, 3, 4],
        )

        head = cfg.get("head", {})
        transform_cfg = cfg.get("transform", {})

        self.model = FCOS(
            backbone=backbone,
            num_classes=cfg["num_classes"],
            min_size=transform_cfg.get("min_size", 800),
            max_size=transform_cfg.get("max_size", 1333),
            score_thresh=head.get("score_thresh", 0.2),
            nms_thresh=head.get("nms_thresh", 0.6),
            detections_per_img=head.get("detections_per_img", 100),
            topk_candidates=head.get("topk_candidates", 1000),
        )
