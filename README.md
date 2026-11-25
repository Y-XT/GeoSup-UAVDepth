# Enhancing Training-Time Geometric Supervision for Self-Supervised Monocular Depth Estimation from UAV Imagery

**Xiaotian Yang, Tomohiro Fukuda** — The University of Osaka

*Under review at the ISPRS Journal of Photogrammetry and Remote Sensing.*

This repository contains the code and the precomputed training triplets for the paper.
The method keeps the MonoViT depth network unchanged and improves the geometric supervision
used during training:

1. **Parallax-guided triplet construction** (paper Section 3.2) selects, for every target frame,
   one backward and one forward source frame whose rotation-compensated parallax is close to a
   preferred value, using local geometry estimated offline by VGGT.
2. **Prior-guided pose refinement** (paper Section 3.3) feeds the VGGT rotation prior (as a
   pixel-level rotation flow) and translation prior (as a 4D direction–magnitude descriptor) into
   PoseNet, which predicts residual corrections around the prior.

Both components are used only during training. Inference uses the depth network alone.

![Overall framework](assets/framework.jpg)

## Repository structure

| Path | Content | Paper |
|---|---|---|
| `triplet_construction/uav_frame_metrics.py` | Lists the frames of every sequence (`frames.jsonl`) used by the builder | Section 3.2 |
| `triplet_construction/vggt_triplet_builder_disp_only.py` | Parallax-guided triplet construction with VGGT | Section 3.2, Eqs. 5–11 |
| `triplets/` | Precomputed triplets and VGGT pose priors for all settings reported in the paper | Tables 1, 9, 10 |
| `methods/networks/pose_flow_encoder.py` | ResNet-18 pose encoder with rotation-flow input (R. Enc.) | Section 3.3, Eqs. 12–13 |
| `methods/networks/pose_tprior_decoder.py` | Translation-prior embedding, feature fusion and residual pose decoder (T. Enc.) | Section 3.3, Eqs. 14–22 |
| `methods/networks/monovit/` | MonoViT depth network (MPViT-S encoder) | Section 4.1.2 |
| `methods/losses/monodepth2/` | Photometric and smoothness losses with minimum reprojection and auto-masking | Section 3.4 |
| `methods/splits/` | Train / validation / test frame lists | Section 4.1.1 |
| `train_tri.py` | Training entry with the final settings for each dataset | Section 4.1.2 |
| `tools/evaluate_uavid2020.py`, `tools/pred_eval_uavula.py` | Prediction and evaluation | Section 4.1.3 |

## Installation

The code was run with Python 3.10, PyTorch 2.8 (CUDA 12.6) and an RTX 4090.

```bash
pip install torch torchvision          # choose the build matching your CUDA version
pip install timm==1.0.19 einops opencv-python scikit-image matplotlib tqdm wandb
pip install mmcv-full==1.7.2 mmsegmentation==0.30.0   # required by the MPViT encoder
```

Download the ImageNet-pretrained MPViT-S weights (`mpvit_small.pth`) from the
[MPViT repository](https://github.com/youngwanLEE/MPViT) and place them at
`pretrained/mpvit_small.pth`, or set `MPVIT_CKPT=/path/to/mpvit_small.pth`.

[VGGT](https://github.com/facebookresearch/vggt) is needed only if you rebuild the triplets yourself
(`pip install -e <path to the vggt repository>`). Training with the precomputed triplets does not need it.

Training logs are written to [Weights & Biases](https://wandb.ai/); run `wandb login` first or set
`WANDB_MODE=offline`.

## Data

Images and reference depths are not redistributed. Please obtain the datasets from their providers:

- **UAVid 2020** (China and Germany subsets): <https://uavid.nl/>. Frames were extracted at 5 fps
  (China) and 10 fps (Germany); obvious static frames were removed.
- **UAV-ula** (Yu et al., 2023), sequences 167–428, at 1 fps.

The training code expects the following layout under `DATA_ROOT` (default `./data`):

```
data/
├── UAVid2020/
│   ├── China/
│   │   ├── Train/seq10/data_1280/0000000000.jpg ...
│   │   ├── Validation/...
│   │   └── Test/UAV_seq1/{images,depth,colmap_masks}/...
│   └── Germany/            # same structure as China
└── UAV-ula/
    ├── Train/DJI_0167/image_02/data/0000000000.jpg ...
    ├── Validation/...
    └── Test/...
```

Sequence folder names and frame file names must match those in `triplets/*/*/triplets.jsonl.gz`
and `methods/splits/`.

## Precomputed triplets

`triplets/<dataset>/tri_win<Nw>_disp<γ>_lap0_rot<θmax>d_U0.5_S0/` contains

- `triplets.jsonl.gz`: one record per target frame with the selected backward / forward source
  frames, VGGT intrinsics (`K_t`, `K_prev`, `K_next`) and the relative pose priors
  (`T_prev_to_t`, `T_next_to_t`) at 518×294;
- `run_config.json`: the exact builder arguments (local paths replaced by `<DATA_ROOT>`);
- `sequences_summary.json`: per-sequence frame and triplet counts.

| Setting | γ | θmax | Used in |
|---|---|---|---|
| `disp0.05_lap0_rot10d` | 0.05 | 10° | **Final setting** (Tables 1–8, 11) |
| `disp0.05_lap0_rot20d` | 0.05 | 20° | Table 9 |
| `disp0.05_lap0_rot1d` | 0.05 | 1° | Table 9 |
| `disp0.03_lap0_rot10d` | 0.03 | 10° | Table 10 |
| `disp0.07_lap0_rot10d` | 0.07 | 10° | Table 10 |

`Nw` is 20 for UAVid-China, 10 for UAVid-Germany and 5 for UAV-ula. Decompress before training:

```bash
find triplets -name 'triplets.jsonl.gz' -exec gunzip -k {} \;
```

### Rebuilding the triplets

```bash
# 1) list the frames of every sequence
python triplet_construction/uav_frame_metrics.py --root_dir <images> --out_dir <metrics>
# 2) parallax-guided selection (defaults are the final paper settings; set --window per dataset)
python triplet_construction/vggt_triplet_builder_disp_only.py \
    --image_root <images> --metrics_root <metrics> --out_root <output> --window 20
```

`<images>` contains one sub-folder per sequence (training and validation sequences together).
Run `--help` to see how every argument maps to the paper. Paper settings: valid parallax 2–80 px,
ρmin = 0.5, θmax = 10°, γ = 0.05, β = 0.35 (fixed in the code), VGGT input 518×294.

## Training

```bash
python train_tri.py --preset china      # or germany / ula
```

The preset sets the final paper configuration: MPViT-S DepthNet, prior-guided ResNet-18 PoseNet
(`--methods MonoViT_VGGT_RFlow_TInj`), 512×288 input, batch size 8, Adam with learning rate 1e-4,
40 epochs, seed 42. Any option given on the command line overrides the preset, e.g.
`--triplet_root triplets/UAV-ula/tri_win5_disp0.05_lap0_rot20d_U0.5_S0` for Table 9.
Checkpoints are saved under `LOG_ROOT` (default `./weights`).

## Evaluation

```bash
# UAVid 2020 (China / Germany); dynamic-object masks are read from colmap_masks/
python tools/evaluate_uavid2020.py --gt_folder data/UAVid2020/China/Test \
    --exp_dir <training run folder> --weights <epoch>

# UAV-ula
python tools/pred_eval_uavula.py --gt_folder data/UAV-ula/Test \
    --exp_dir <training run folder> --weights <epoch>
```

Both scripts apply per-image median scaling and evaluate within the 5th–95th percentile of the valid
reference depth of each image, and report AbsRel, SqRel, RMSE, RMSElog and δ < 1.25, 1.25², 1.25³.

## Citation

The citation will be added after publication.

## License and acknowledgements

This code is built on [Monodepth2](https://github.com/nianticlabs/monodepth2) and is released under
the Monodepth2 license (non-commercial use only, see [LICENSE](LICENSE)).
We also thank the authors of [MonoViT](https://github.com/zxcqlf/MonoViT),
[MPViT](https://github.com/youngwanLEE/MPViT), [VGGT](https://github.com/facebookresearch/vggt)
and [GasMono](https://github.com/zxcqlf/GasMono) for releasing their code.
The pose priors in `triplets/` were produced with the VGGT model; please also follow the VGGT license
when using them.
