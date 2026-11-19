"""Colab/local CLI entry point to reproduce G0 2x2 double dissociation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from srwm.env_g0 import DEFAULT_G0_CFG, evaluate_cem_planning, generate_dataset
from srwm.models import FactoredWorldModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run G0 2x2 double dissociation")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--out", type=Path, default=Path("outputs/g0_results.json"))
    args = parser.parse_args()

    cfg = dict(DEFAULT_G0_CFG, grad_steps=args.steps, batch_size=256, k_train=3, lr=1e-3, l1_gate_weight=5e-4)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running SR-WM G0 on {device} | steps={args.steps} | seeds={seeds}")

    eval_ds = {
        m: generate_dataset(m, 128, cfg["ep_len"], 101 + i * 101, cfg)
        for i, m in enumerate(["id_center", "avoid", "anti_goal", "multi_3"])
    }
    specs = [
        ("M0_Holistic_Global", False, False),
        ("M1_Sparse_Global", True, False),
        ("M2_Dense_Relational", False, True),
        ("M3_SR_WM_Ours", True, True),
    ]

    runs = {name: [] for name, _, _ in specs}
    for seed in seeds:
        tr_ds = generate_dataset("id_center", 500, cfg["ep_len"], seed + 10, cfg)
        states_t = torch.from_numpy(tr_ds["states"]).to(device)
        actions_t = torch.from_numpy(tr_ds["actions"]).to(device)
        n_ep, ep_len, _ = actions_t.shape

        for name, sp, rel in specs:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed + 500)
            model = FactoredWorldModel(128, sp, rel).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["grad_steps"])

            t0 = time.perf_counter()
            model.train()
            for _ in range(cfg["grad_steps"]):
                ep_i = torch.from_numpy(rng.integers(0, n_ep, size=cfg["batch_size"])).to(device)
                st_i = torch.from_numpy(rng.integers(0, ep_len - cfg["k_train"] + 1, size=cfg["batch_size"])).to(device)
                z = states_t[ep_i, st_i]
                l_mse, l_g = 0.0, 0.0
                for k in range(cfg["k_train"]):
                    z, g_l1 = model.forward_step(z, actions_t[ep_i, st_i + k])
                    l_mse = l_mse + F.mse_loss(z, states_t[ep_i, st_i + k + 1])
                    l_g = l_g + g_l1
                loss = (l_mse / cfg["k_train"]) + (cfg["l1_gate_weight"] * (l_g / cfg["k_train"]) if sp else 0.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
            dt = time.perf_counter() - t0

            cem_av = evaluate_cem_planning(model, "avoid", cfg, seed, device)
            cem_an = evaluate_cem_planning(model, "anti_goal", cfg, seed, device)
            runs[name].append({"seed": seed, "sec": round(dt, 1), "cem_avoid": cem_av, "cem_anti": cem_an})
            print(f"[{name:20s} s={seed}] ({dt:4.1f}s) | CEM Avoid={cem_av['success_rate']:.2f} | CEM Anti={cem_an['success_rate']:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
