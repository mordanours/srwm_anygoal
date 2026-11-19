"""G1b: Train SR-WM-Rigid & M0-M2 on official AnyGoal PushT state data and run rollout + CEM planning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from srwm.pusht_models import PushTWorldModel, angle_diff_rad, pusht_state_loss


def load_state_file(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        d = np.load(path)
        return {k: d[k] for k in ("state", "action", "ep_offset", "ep_len")}
    with h5py.File(path, "r") as f:
        return {
            "state": f["state"][:].astype(np.float32),
            "action": f["action"][:].astype(np.float32),
            "ep_offset": f["ep_offset"][:].astype(np.int64),
            "ep_len": f["ep_len"][:].astype(np.int32),
        }


def build_windows(ds: dict[str, np.ndarray], ep_slice: slice, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    offsets, lens = ds["ep_offset"][ep_slice], ds["ep_len"][ep_slice]
    s_all, a_all = ds["state"], ds["action"]
    s_wins, a_wins = [], []
    for off, length in zip(offsets, lens):
        if length <= horizon:
            continue
        for t in range(0, length - horizon):
            s_wins.append(s_all[off + t : off + t + horizon + 1])
            a_wins.append(a_all[off + t : off + t + horizon])
    return np.stack(s_wins, axis=0), np.stack(a_wins, axis=0)


@torch.no_grad()
def eval_rollout(model: PushTWorldModel, s_np: np.ndarray, a_np: np.ndarray, h: int, device: torch.device) -> dict:
    model.eval()
    s_gt = torch.from_numpy(s_np).to(device)
    a_gt = torch.from_numpy(a_np).to(device)
    s_cur = s_gt[:, 0]
    losses, blk_errs, ang_errs, gates = [], [], [], []
    for t in range(h):
        s_cur, gm = model.forward_step(s_cur, a_gt[:, t], hard_gate_thresh=0.05)
        losses.append(float(pusht_state_loss(s_cur, s_gt[:, t + 1]).item()))
        blk_errs.append(float(torch.norm(s_cur[:, 2:4] - s_gt[:, t + 1, 2:4], dim=-1).mean().item()))
        ang_errs.append(float(torch.abs(angle_diff_rad(s_cur[:, 4:5], s_gt[:, t + 1, 4:5])).mean().item()))
        gates.append(float(gm.item()))
    return {
        "norm_mse": round(float(np.mean(losses)), 6),
        "blk_err_px": round(float(np.mean(blk_errs)), 2),
        "ang_err_rad": round(float(np.mean(ang_errs)), 4),
        "blk_drift_px": round(float(torch.norm(s_cur[:, 2:4] - s_gt[:, 0, 2:4], dim=-1).mean().item()), 2),
        "mean_gate": round(float(np.mean(gates)), 4),
    }


@torch.no_grad()
def eval_cem_anygoal(model: PushTWorldModel, env_id: str, n_eps: int, device: torch.device, seed: int = 0) -> dict:
    """Closed-loop CEM planning inside real Pymunk PushT/AvoidT/PushTAntigoal env."""
    import anygoal  # noqa: F401
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    model.eval()
    env = gym.make(env_id)
    h_plan, pop, elites, iters = 10, 128, 16, 3
    dists, successes = [], []

    for ep in range(n_eps):
        obs, info = env.reset(seed=seed * 100 + ep)
        goal_pos = np.asarray(info["goal_pose"][:2], dtype=np.float32)
        goal_t = torch.from_numpy(goal_pos).to(device)
        collided = False

        for _ in range(40):
            s_cur = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).to(device).unsqueeze(0)
            mu = torch.zeros(h_plan, 2, device=device)
            std = torch.ones(h_plan, 2, device=device) * 0.7

            for _ in range(iters):
                eps_n = torch.randn(pop, h_plan, 2, device=device)
                acts = torch.clamp(mu.unsqueeze(0) + std.unsqueeze(0) * eps_n, -1.0, 1.0)
                z = s_cur.expand(pop, 7)
                cost = torch.zeros(pop, device=device)
                for tau in range(h_plan):
                    z, _ = model.forward_step(z, acts[:, tau, :], hard_gate_thresh=0.05)
                    if "AvoidT" in env_id:
                        d_goal = torch.norm(z[:, 0:2] - goal_t, dim=-1) / 512.0
                        d_blk = torch.norm(z[:, 0:2] - z[:, 2:4], dim=-1)
                        cost = cost + d_goal + torch.relu(65.0 - d_blk) * 0.1
                    else:
                        d_blk_goal = torch.norm(z[:, 2:4] - goal_t, dim=-1) / 512.0
                        d_agt_blk = torch.norm(z[:, 0:2] - z[:, 2:4], dim=-1) / 512.0
                        cost = cost + 2.5 * d_blk_goal + 0.4 * d_agt_blk
                top_idx = torch.topk(-cost, k=elites).indices
                mu = acts[top_idx].mean(dim=0)
                std = acts[top_idx].std(dim=0) + 0.05

            obs, _, term, trunc, info = env.step(mu[0].cpu().numpy())
            if info.get("n_contacts", 0) > 0:
                collided = True
            if term or trunc:
                break

        s_end = np.asarray(obs["state"], dtype=np.float32)
        if "AvoidT" in env_id:
            d_fin = float(np.linalg.norm(s_end[0:2] - goal_pos))
            successes.append((d_fin < 35.0) and (not collided))
        else:
            d_fin = float(np.linalg.norm(s_end[2:4] - goal_pos))
            successes.append(d_fin < 45.0)
        dists.append(d_fin)

    env.close()
    return {"final_dist_px": round(float(np.mean(dists)), 2), "success_rate": round(float(np.mean(successes)), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description="G1b Official PushT State Benchmark")
    parser.add_argument("--train-file", type=Path, required=True, help="Official pusht_expert_train_state.npz (or .h5)")
    parser.add_argument("--test-dir", type=Path, required=True, help="Dir containing avoidt/antigoal/stackt (.npz or .h5)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--run-cem", action="store_true", help="Also run closed-loop CEM planning in AnyGoal envs")
    parser.add_argument("--out", type=Path, default=Path("outputs/g1b_pusht_results.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tr_ds = load_state_file(args.train_file)
    n_eps = int(tr_ds["ep_len"].shape[0])
    n_tr_eps = max(1, int(0.9 * n_eps))
    s_tr, a_tr = build_windows(tr_ds, slice(0, n_tr_eps), horizon=3)
    s_id, a_id = build_windows(tr_ds, slice(n_tr_eps, None), horizon=15)

    rng_ev = np.random.default_rng(12345)
    eval_sets = {}
    for k, arrs in [("id_pusht", (s_id, a_id))] + [
        (name, build_windows(load_state_file(p), slice(None), 15))
        for name, stem in [("avoidt", "avoidt_test"), ("antigoal", "pusht_antigoal_test"), ("stackt", "stackt_test")]
        for p in [args.test_dir / f"{stem}_state.npz" if (args.test_dir / f"{stem}_state.npz").is_file() else args.test_dir / f"{stem}.h5"]
        if p.is_file()
    ]:
        idx = rng_ev.choice(arrs[0].shape[0], size=min(256, arrs[0].shape[0]), replace=False)
        eval_sets[k] = (arrs[0][idx], arrs[1][idx])

    specs = [
        ("M0_Global_MLP", False, False),
        ("M1_Sparse_Global", True, False),
        ("M2_Dense_BodyFrame", False, True),
        ("M3_SR_WM_Rigid", True, True),
    ]
    results = {name: [] for name, _, _ in specs}

    for seed in seeds:
        for name, sp, bf in specs:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed + 500)
            model = PushTWorldModel(128, sp, bf).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
            s_t, a_t = torch.from_numpy(s_tr).to(device), torch.from_numpy(a_tr).to(device)

            t0 = time.perf_counter()
            model.train()
            for _ in range(args.steps):
                idx = torch.from_numpy(rng.integers(0, s_t.shape[0], size=256)).to(device)
                sc, lp, lg = s_t[idx, 0], 0.0, 0.0
                for k in range(3):
                    sc, gm = model.forward_step(sc, a_t[idx, k])
                    lp = lp + pusht_state_loss(sc, s_t[idx, k + 1])
                    lg = lg + gm
                loss = (lp / 3.0) + (5e-4 * (lg / 3.0) if sp else 0.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
            dt = time.perf_counter() - t0

            ev = {k: eval_rollout(model, *v, h=15, device=device) for k, v in eval_sets.items()}
            cem = {}
            if args.run_cem:
                cem["avoidt"] = eval_cem_anygoal(model, "anygoal/AvoidT-v1", 20, device, seed)
                cem["antigoal"] = eval_cem_anygoal(model, "anygoal/PushTAntigoal-v3", 20, device, seed)
            results[name].append({"seed": seed, "sec": round(dt, 1), "eval": ev, "cem": cem})
            print(f"[{name:18s} s={seed}] ({dt:4.1f}s) | " + " | ".join(f"{k}={v['blk_err_px']}px/{v['ang_err_rad']}rad" for k, v in ev.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
