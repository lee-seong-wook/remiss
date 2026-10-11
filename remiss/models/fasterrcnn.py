from __future__ import annotations

from torchvision.models.detection import FasterRCNN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .base import BaseDetectionWrapper, build_backbone_with_fpn


class FasterRCNNWrapper(BaseDetectionWrapper):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        backbone = build_backbone_with_fpn(
            cfg,
            extra_blocks=LastLevelMaxPool(),
        )

        rpn = cfg.get("rpn", {})
        roi = cfg.get("roi_head", {})
        transform_cfg = cfg.get("transform", {})

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=cfg["num_classes"],
            min_size=transform_cfg.get("min_size", 800),
            max_size=transform_cfg.get("max_size", 1333),
            rpn_pre_nms_top_n_train=rpn.get("pre_nms_top_n_train", 2000),
            rpn_pre_nms_top_n_test=rpn.get("pre_nms_top_n_test", 1000),
            rpn_post_nms_top_n_train=rpn.get("post_nms_top_n_train", 2000),
            rpn_post_nms_top_n_test=rpn.get("post_nms_top_n_test", 1000),
            rpn_nms_thresh=rpn.get("nms_thresh", 0.7),
            rpn_score_thresh=rpn.get("score_thresh", 0.0),
            rpn_fg_iou_thresh=rpn.get("fg_iou_thresh", 0.7),
            rpn_bg_iou_thresh=rpn.get("bg_iou_thresh", 0.3),
            rpn_batch_size_per_image=rpn.get("batch_size_per_image", 256),
            rpn_positive_fraction=rpn.get("positive_fraction", 0.5),
            box_score_thresh=roi.get("box_score_thresh", 0.05),
            box_nms_thresh=roi.get("box_nms_thresh", 0.5),
            box_detections_per_img=roi.get("box_detections_per_img", 100),
            box_fg_iou_thresh=roi.get("box_fg_iou_thresh", 0.5),
            box_bg_iou_thresh=roi.get("box_bg_iou_thresh", 0.5),
            box_batch_size_per_image=roi.get("box_batch_size_per_image", 512),
            box_positive_fraction=roi.get("box_positive_fraction", 0.25),
        )
