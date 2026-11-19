# `srwm_anygoal` — Sparse-Regime Relational World Models (SR-WM)

面向 **`AnyGoal (arXiv:2610.26844)`** 组合泛化基准的最小可安装代码仓库（支持本地 Windows 与 Google Colab Linux 一键安装）。

## 目录结构

```text
srwm_anygoal/
├── pyproject.toml                   # pip / uv 一键安装配置（核心依赖极轻）
├── srwm/
│   ├── __init__.py
│   ├── models.py                    # SR-WM（稀疏模式门控 + SE(2) 局部接触标架）与 M0–M3 消融
│   └── env_g0.py                    # G0 2D 推块物理环境与 CEM 规划器
├── scripts/
│   ├── run_g0.py                    # G0 双解离一键复现脚本
│   └── extract_anygoal_states.py    # 剔除 AnyGoal .h5 中的 RGB 像素，压成几十 MB 纯状态 .npz
└── notebooks/
    └── colab_quickstart.ipynb       # Google Colab 一键挂盘、安装、抽状态与训练模板
```

## 快速安装

### 本地（或云端 Linux / Colab）
```bash
pip install -e ./srwm_anygoal
# 若在 Linux / Colab 上同时安装官方 AnyGoal 基准：
pip install -e "./srwm_anygoal[anygoal]"
```

### 一键运行 G0 验证
```bash
python srwm_anygoal/scripts/run_g0.py --steps 3000 --seeds 0,1,2
```

## Colab 省空间工作流建议
1. 打开 `notebooks/colab_quickstart.ipynb`，挂载 Google Drive 仅用于保存**几十 MB 的纯状态 `.npz`、实验 JSON 和模型 `.pt` 权重**；
2. 12.5 GB 的 `pusht_expert_train.h5.zst` 在 Colab 临时盘 `/content/tmp_anygoal` 下载解压，用 `scripts/extract_anygoal_states.py` 抽完纯状态存回 Drive 后直接删除原视频包。
