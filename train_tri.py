# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
from utils import set_global_seed
import sys
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

# 论文最终设置（Section 4.1.2）：三个数据集只差数据路径、split 与选帧窗口 Nw。
# 用法：python train_tri.py --preset {china,germany,ula} [其它 MonodepthOptions 参数]
# 数据根目录与权重输出目录分别由环境变量 DATA_ROOT、LOG_ROOT 指定（默认 ./data、./weights）。
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO_ROOT, "data"))
LOG_ROOT = os.environ.get("LOG_ROOT", os.path.join(REPO_ROOT, "weights"))

PRESETS = {
    "china": dict(project_name="UAVid_China", split="UAVid2020_China", dataset="UAVid_TriDataset",
                  data_path=os.path.join(DATA_ROOT, "UAVid2020", "China"),
                  triplet_root=os.path.join(REPO_ROOT, "triplets", "UAVid-China",
                                            "tri_win20_disp0.05_lap0_rot10d_U0.5_S0")),
    "germany": dict(project_name="UAVid_Germany", split="UAVid2020_Germany", dataset="UAVid_TriDataset",
                    data_path=os.path.join(DATA_ROOT, "UAVid2020", "Germany"),
                    triplet_root=os.path.join(REPO_ROOT, "triplets", "UAVid-Germany",
                                              "tri_win10_disp0.05_lap0_rot10d_U0.5_S0")),
    "ula": dict(project_name="UAVula_R1", split="UAVula_R1", dataset="UAVula_TriDataset",
                data_path=os.path.join(DATA_ROOT, "UAV-ula"),
                triplet_root=os.path.join(REPO_ROOT, "triplets", "UAV-ula",
                                          "tri_win5_disp0.05_lap0_rot10d_U0.5_S0")),
}


def preset_argv(name):
    p = PRESETS[name]
    return [
        "--project_name", p["project_name"],
        "--methods", "MonoViT_VGGT_RFlow_TInj",  # 论文方法：MonoViT + 旋转流编码（R. Enc.）+ 平移先验注入（T. Enc.）
        "--log_dir", os.path.join(LOG_ROOT, p["project_name"]),
        "--model_name", "",
        "--split", p["split"],
        "--dataset", p["dataset"],
        "--data_path", p["data_path"],
        "--triplet_root", p["triplet_root"],
        "--batch_size", "8",
        "--learning_rate", "1e-4",
        "--num_workers", "16",
        "--frame_ids", "0", "-1", "1",
        "--height", "288",
        "--width", "512",
        "--num_epochs", "40",
        "--scheduler_step_size", "20",
        "--max_depth", "150.0",
        "--log_frequency", "300",
        "--scales", "0", "1", "2", "3",

        # automask 可视化与统计（论文表 8、图 7–8、图 10 的来源）
        "--enable_automask_margin_viz",
        "--automask_margin_viz_samples", "1",
        "--enable_debug_metrics",
        "--enable_flip",
    ]


def inject_preset():
    """把 --preset 展开成预设参数，插在用户参数之前，使命令行显式给出的参数优先。"""
    if "--preset" not in sys.argv:
        raise SystemExit("usage: python train_tri.py --preset {china,germany,ula} [extra options]")
    i = sys.argv.index("--preset")
    name = sys.argv[i + 1]
    if name not in PRESETS:
        raise SystemExit(f"--preset must be one of {sorted(PRESETS)}")
    del sys.argv[i:i + 2]
    sys.argv[1:1] = preset_argv(name)


def main():
    inject_preset()
    options = MonodepthOptions()
    # 解析命令行参数并生成最终的 model_name。
    opts = options.parse()  # ★ 自动命名在 options.parse() 内完成
    if getattr(opts, "seed", None) is not None and int(opts.seed) >= 0:
        set_global_seed(int(opts.seed), deterministic=True)
        print(f"[INFO] Seed:        {int(opts.seed)} (cudnn deterministic)")
    else:
        print("[INFO] Seed:        disabled")

    print(f"[INFO] Project:      {opts.project_name}")
    print(f"[INFO] Method:       {opts.methods}")
    print(f"[INFO] Model name:   {opts.model_name}")

    # 启动标准训练循环；Trainer 内部已支持三元组数据。
    trainer = Trainer(opts)
    trainer.train()

if __name__ == "__main__":
    main()
