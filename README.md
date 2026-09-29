# ReMiss: Missed-Object History for Data Scheduling in Driving-Scene Object Detection

**Computers, Materials & Continua (2026)**  
Seongwook Lee, Seungyeong Kim, Jaemin Oh, L. Minh Dang, Hyeonjoon Moon  
DOI: [10.32604/cmc.2026.087886](https://doi.org/10.32604/cmc.2026.087886)  
Paper: https://www.techscience.com/cmc/online/detail/28371

Experiment code for **ReMiss**, a training-time data scheduling method that uses missed-object history as the scheduling signal for driving-scene object detection, without changing the detector architecture or loss.

## Overview

ReMiss records each ground-truth instance's miss status and miss-count history (MissBank) across training epochs. Hard Replay then prioritizes full images containing currently missed objects and schedules them into later epochs under a replay budget. Evaluated on KITTI, BDD100K, Cityscapes, and nuImages with FCOS, Faster R-CNN, and DINO.

## Paper

If you use this repository, please cite the paper below.

```bibtex
@article{lee2026remiss,
  title   = {ReMiss: Missed-Object History for Data Scheduling in Driving-Scene Object Detection},
  author  = {Lee, Seongwook and Kim, Seungyeong and Oh, Jaemin and Dang, L. Minh and Moon, Hyeonjoon},
  journal = {Computers, Materials \& Continua},
  year    = {2026},
  doi     = {10.32604/cmc.2026.087886}
}
```

## Repository Structure

```text
remiss/
|-- configs/
|-- docs/
|-- figures/
|-- scripts/
|-- src/
|-- THIRD_PARTY_NOTICES.md
|-- README.md
`-- requirements.txt
```

## Status

Experiment code is being prepared for release. Training/evaluation scripts and configs will be added here as the cleanup progresses.
