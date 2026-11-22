"""G1c: Frozen LFS-WM + Anti-Collapse Residual-Anchored Loss + Stratified Eval & CEM."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from srwm.lfs_wm import (
    LFSWorldModel,
    angle_diff_rad,
    pusht_state_error,
    residual_anchored_loss,
)


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


def build_macro_windows(
    ds: dict[str, np.ndarray],
    ep_slice: slice,
    history_length: int = 3,
    rollout_steps: int = 3,
    frame_skip: int = 5,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build macro-step windows matching AnyGoal RolloutProtocol (frame_skip=5, H_hist=3).

    Returns:
        s_frames: (N, history_length + rollout_steps, 7)
        a_blocks: (N, history_length - 1 + rollout_steps, frame_skip, 2)
    """
    offsets, lens = ds["ep_offset"][ep_slice], ds["ep_len"][ep_slice]
    s_all, a_all = ds["state"], ds["action"]
    req_len = (history_length + rollout_steps - 1) * frame_skip + 1
    s_offs = np.arange(history_length + rollout_steps) * frame_skip
    a_starts = np.arange(history_length - 1 + rollout_steps) * frame_skip
    a_offs = a_starts[:, None] + np.arange(frame_skip)[None, :]

    s_wins, a_wins = [], []
    for off, length in zip(offsets, lens):
        if length < req_len:
            continue
        starts = np.arange(0, length - req_len + 1, stride)
        s_wins.append(s_all[off + starts[:, None] + s_offs[None, :]])
        a_wins.append(a_all[off + starts[:, None, None] + a_offs[None, :, :]])
    return np.concatenate(s_wins, axis=0), np.concatenate(a_wins, axis=0)


@torch.no_grad()
def eval_static_block(s_frames: np.ndarray, history_length: int = 3) -> dict:
    """Compute trivial zero-motion block baseline (delta_b == 0) on the same windows."""
    s0 = s_frames[:, history_length - 1 : history_length, :]
    gt = s_frames[:, history_length:, :]
    blk_errs = np.linalg.norm(gt[:, :, 2:4] - s0[:, :, 2:4], axis=-1).mean(axis=1)
    ang_errs = np.abs(
        np.arctan2(np.sin(gt[:, :, 4] - s0[:, :, 4]), np.cos(gt[:, :, 4] - s0[:, :, 4]))
    ).mean(axis=1)
    disp_h = np.linalg.norm(gt[:, -1, 2:4] - s0[:, 0, 2:4], axis=-1)
    mov = disp_h > 5.0
    return {
        "blk_err_px": round(float(blk_errs.mean()), 2),
        "ang_err_rad": round(float(ang_errs.mean()), 4),
        "mov_frac": round(float(mov.mean()), 4),
        "mov_blk_err_px": round(float(blk_errs[mov].mean()) if mov.any() else 0.0, 2),
        "mov_ang_err_rad": round(float(ang_errs[mov].mean()) if mov.any() else 0.0, 4),
    }


@torch.no_grad()
def eval_rollout_stratified(
    model: LFSWorldModel,
    s_np: np.ndarray,
    a_np: np.ndarray,
    device: torch.device,
    history_length: int = 3,
    rollout_steps: int = 3,
) -> dict:
    """Anti-shortcut stratified rollout eval: All-Windows + Moving-Only + Gate Recall/Sparsity."""
    model.eval()
    s_gt = torch.from_numpy(s_np).to(device)
    a_gt = torch.from_numpy(a_np).to(device)

    s_hist = s_gt[:, :history_length]
    s_cur0 = s_hist[:, -1]
    step_mses, step_blk_errs, step_ang_errs = [], [], []
    step_gates_raw = []

    for k in range(rollout_steps):
        a_hist = a_gt[:, k : k + history_length]
        pred_s, aux = model.forward_step(s_hist, a_hist, use_ste=True)
        gt_next = s_gt[:, history_length + k]
        step_mses.append(pusht_state_error(pred_s, gt_next))
        step_blk_errs.append(torch.norm(pred_s[:, 2:4] - gt_next[:, 2:4], dim=-1))
        step_ang_errs.append(torch.abs(angle_diff_rad(pred_s[:, 4:5], gt_next[:, 4:5])).squeeze(-1))
        step_gates_raw.append(aux["g_ab_blk"].squeeze(-1))
        s_hist = torch.cat([s_hist[:, 1:], pred_s.unsqueeze(1)], dim=1)

    win_mse = torch.stack(step_mses, dim=1).mean(dim=1)
    win_blk = torch.stack(step_blk_errs, dim=1).mean(dim=1)
    win_ang = torch.stack(step_ang_errs, dim=1).mean(dim=1)
    blk_drift = torch.norm(s_hist[:, -1, 2:4] - s_cur0[:, 2:4], dim=-1)

    disp_h = torch.norm(s_gt[:, -1, 2:4] - s_cur0[:, 2:4], dim=-1)
    mov_win = disp_h > 5.0

    # Gate diagnostics on step k=0 ground-truth block motion (>1.5px or >0.01rad)
    gt_dp0 = torch.norm(s_gt[:, history_length, 2:4] - s_cur0[:, 2:4], dim=-1)
    gt_dth0 = torch.abs(angle_diff_rad(s_gt[:, history_length, 4:5], s_cur0[:, 4:5])).squeeze(-1)
    is_mov0 = (gt_dp0 > 1.5) | (gt_dth0 > 0.01)
    g0 = step_gates_raw[0]
    gate_recall = float((g0[is_mov0] >= 0.5).float().mean().item()) if is_mov0.any() else 1.0
    gate_sparsity = float((g0[~is_mov0] < 0.5).float().mean().item()) if (~is_mov0).any() else 1.0

    return {
        "norm_mse": round(float(win_mse.mean().item()), 6),
        "blk_err_px": round(float(win_blk.mean().item()), 2),
        "ang_err_rad": round(float(win_ang.mean().item()), 4),
        "blk_drift_px": round(float(blk_drift.mean().item()), 2),
        "mov_frac": round(float(mov_win.float().mean().item()), 4),
        "mov_blk_err_px": round(float(win_blk[mov_win].mean().item()) if mov_win.any() else 0.0, 2),
        "mov_ang_err_rad": round(float(win_ang[mov_win].mean().item()) if mov_win.any() else 0.0, 4),
        "mean_gate": round(float(torch.stack(step_gates_raw, dim=1).mean().item()), 4),
        "gate_recall": round(gate_recall, 4),
        "gate_sparsity": round(gate_sparsity, 4),
    }


@torch.no_grad()
def check_free_space_zero_leakage(model: LFSWorldModel, device: torch.device) -> float:
    """Assert G1c-3: moving far-away T-block causes strictly 0.0 change to agent prediction."""
    model.eval()
    s1 = torch.tensor([[200.0, 200.0, 400.0, 400.0, 0.5, 10.0, -5.0]], device=device).unsqueeze(1).repeat(1, 3, 1)
    s2 = s1.clone()
    s2[:, :, 2:4] = torch.tensor([120.0, 410.0], device=device)  # far from agent & walls
    #Wait: keep block far from agent (200,200) and walls (>=64px -> in [71, 440])
    s1[:, :, 2:4] = torch.tensor([400.0, 400.0], device=device)
    s2[:, :, 2:4] = torch.tensor([380.0, 100.0], device=device)
    a_hist = torch.full((1, 3, model.frame_skip, 2), 0.3, device=device)
    p1, _ = model.forward_step(s1, a_hist, use_ste=True)
    p2, _ = model.forward_step(s2, a_hist, use_ste=True)
    return float(torch.max(torch.abs(p1[:, [0, 1, 5, 6]] - p2[:, [0, 1, 5, 6]])).item())


@torch.no_grad()
def eval_cem_anygoal(
    model: LFSWorldModel,
    env_id: str,
    n_eps: int,
    device: torch.device,
    seed: int = 0,
    frame_skip: int = 5,
    history_length: int = 3,
) -> dict:
    """Closed-loop macro-step CEM planning inside real Pymunk AnyGoal environments."""
    import anygoal  # noqa: F401
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    model.eval()
    env = gym.make(env_id)
    h_plan, pop, elites, iters = 5, 128, 16, 3
    max_macro_steps = 10  # 10 * 5 = 50 env steps (matches AnyGoal eval_budget=50)
    dists, successes = [], []

    for ep in range(n_eps):
        obs, info = env.reset(seed=seed * 100 + ep)
        if "StackT" in env_id and "stack_block_target_pose" in info:
            target_pose = np.asarray(info["stack_block_target_pose"], dtype=np.float32)
        else:
            target_pose = np.asarray(info["goal_pose"], dtype=np.float32)
        goal_pos_t = torch.from_numpy(target_pose[:2]).to(device)
        goal_th_t = torch.tensor([[float(target_pose[2])]], device=device) if len(target_pose) >= 3 else None

        s0 = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).to(device)
        s_q: deque[torch.Tensor] = deque([s0] * history_length, maxlen=history_length)
        a_q: deque[torch.Tensor] = deque(
            [torch.zeros(frame_skip, 2, device=device)] * (history_length - 1),
            maxlen=history_length - 1,
        )
        collided = False

        for _ in range(max_macro_steps):
            s_hist_cur = torch.stack(list(s_q), dim=0).unsqueeze(0).expand(pop, history_length, 7)
            a_hist_prev = torch.stack(list(a_q), dim=0).unsqueeze(0).expand(pop, history_length - 1, frame_skip, 2)

            mu = torch.zeros(h_plan, frame_skip, 2, device=device)
            std = torch.ones(h_plan, frame_skip, 2, device=device) * 0.7

            for _ in range(iters):
                eps_n = torch.randn(pop, h_plan, frame_skip, 2, device=device)
                acts = torch.clamp(mu.unsqueeze(0) + std.unsqueeze(0) * eps_n, -1.0, 1.0)
                sh = s_hist_cur.clone()
                ah = a_hist_prev
                cost = torch.zeros(pop, device=device)
                for tau in range(h_plan):
                    a_in = torch.cat([ah[:, -(history_length - 1) :], acts[:, tau : tau + 1]], dim=1)
                    z, _ = model.forward_step(sh, a_in, use_ste=True)
                    sh = torch.cat([sh[:, 1:], z.unsqueeze(1)], dim=1)
                    ah = a_in
                    if "AvoidT" in env_id:
                        d_goal = torch.norm(z[:, 0:2] - goal_pos_t, dim=-1) / 512.0
                        d_blk = torch.norm(z[:, 0:2] - z[:, 2:4], dim=-1)
                        cost = cost + d_goal + torch.relu(65.0 - d_blk) * 0.1
                    else:
                        d_blk_goal = torch.norm(z[:, 2:4] - goal_pos_t, dim=-1) / 512.0
                        d_agt_blk = torch.norm(z[:, 0:2] - z[:, 2:4], dim=-1) / 512.0
                        cost = cost + 2.5 * d_blk_goal + 0.4 * d_agt_blk
                        if "StackT" in env_id and goal_th_t is not None:
                            cost = cost + 0.5 * torch.abs(angle_diff_rad(z[:, 4:5], goal_th_t)).squeeze(-1) / np.pi
                top_idx = torch.topk(-cost, k=elites).indices
                mu = acts[top_idx].mean(dim=0)
                std = acts[top_idx].std(dim=0) + 0.05

            exec_block = mu[0].cpu().numpy()
            done = False
            for f in range(frame_skip):
                obs, _, term, trunc, info = env.step(exec_block[f])
                if info.get("n_contacts", 0) > 0:
                    collided = True
                if term or trunc:
                    done = True
                    break
            s_q.append(torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).to(device))
            a_q.append(mu[0].detach())
            if done:
                break

        s_end = np.asarray(obs["state"], dtype=np.float32)
        if "AvoidT" in env_id:
            d_fin = float(np.linalg.norm(s_end[0:2] - target_pose[:2]))
            successes.append((d_fin < 35.0) and (not collided))
        else:
            d_fin = float(np.linalg.norm(s_end[2:4] - target_pose[:2]))
            successes.append(d_fin < 45.0)
        dists.append(d_fin)

    env.close()
    return {"final_dist_px": round(float(np.mean(dists)), 2), "success_rate": round(float(np.mean(successes)), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description="G1c Official PushT LFS-WM Benchmark")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--run-cem", action="store_true")
    parser.add_argument("--cem-eps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/g1c_pusht_results.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tr_ds = load_state_file(args.train_file)
    n_eps = int(tr_ds["ep_len"].shape[0])
    n_tr_eps = max(1, int(0.9 * n_eps))

    s_tr, a_tr = build_macro_windows(tr_ds, slice(0, n_tr_eps), 3, 3, 5, stride=1)
    s_id, a_id = build_macro_windows(tr_ds, slice(n_tr_eps, None), 3, 3, 5, stride=2)

    rng_ev = np.random.default_rng(12345)
    eval_sets = {}
    for k, arrs in [("id_pusht", (s_id, a_id))] + [
        (name, build_macro_windows(load_state_file(p), slice(None), 3, 3, 5, stride=1))
        for name, stem in [("avoidt", "avoidt_test"), ("antigoal", "pusht_antigoal_test"), ("stackt", "stackt_test")]
        for p in [args.test_dir / f"{stem}_state.npz" if (args.test_dir / f"{stem}_state.npz").is_file() else args.test_dir / f"{stem}.h5"]
        if p.is_file()
    ]:
        n_pick = min(1024, arrs[0].shape[0])
        idx = rng_ev.choice(arrs[0].shape[0], size=n_pick, replace=False)
        eval_sets[k] = (arrs[0][idx], arrs[1][idx])

    static_baseline = {k: eval_static_block(v[0], history_length=3) for k, v in eval_sets.items()}
    print("[StaticBlock] " + " | ".join(f"{k}=All:{v['blk_err_px']}px/Mov:{v['mov_blk_err_px']}px" for k, v in static_baseline.items()))

    specs = [
        ("M0_Global_MLP", False, False),
        ("M1_Sparse_Global", True, False),
        ("M2_Dense_LFS", False, True),
        ("M3_LFS_WM", True, True),
    ]
    results = {"StaticBlock": static_baseline, **{name: [] for name, _, _ in specs}}
    warmup_steps = int(0.15 * args.steps)

    for seed in seeds:
        for name, sp, bf in specs:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed + 500)
            model = LFSWorldModel(128, 3, 5, sp, bf).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
            s_t, a_t = torch.from_numpy(s_tr).to(device), torch.from_numpy(a_tr).to(device)

            t0 = time.perf_counter()
            model.train()
            for step in range(args.steps):
                idx = torch.from_numpy(rng.integers(0, s_t.shape[0], size=args.batch_size)).to(device)
                s_hist = s_t[idx, :3]
                use_ste = step >= warmup_steps
                loss = 0.0
                for k in range(3):
                    a_hist = a_t[idx, k : k + 3]
                    pred_s, aux = model.forward_step(s_hist, a_hist, use_ste=use_ste)
                    loss = loss + residual_anchored_loss(pred_s, s_t[idx, 3 + k], aux, use_sparse_gate=sp)
                    s_hist = torch.cat([s_hist[:, 1:], pred_s.unsqueeze(1)], dim=1)
                loss = loss / 3.0
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
            dt = time.perf_counter() - t0

            ev = {k: eval_rollout_stratified(model, *v, device=device) for k, v in eval_sets.items()}
            leak = check_free_space_zero_leakage(model, device) if (sp and bf) else None
            cem = {}
            if args.run_cem:
                cem["avoidt"] = eval_cem_anygoal(model, "anygoal/AvoidT-v1", args.cem_eps, device, seed)
                cem["antigoal"] = eval_cem_anygoal(model, "anygoal/PushTAntigoal-v3", args.cem_eps, device, seed)
                cem["stackt"] = eval_cem_anygoal(model, "anygoal/StackT-v2", args.cem_eps, device, seed)
            results[name].append({"seed": seed, "sec": round(dt, 1), "leak": leak, "eval": ev, "cem": cem})
            print(
                f"[{name:16s} s={seed}] ({dt:4.1f}s) | "
                + " | ".join(
                    f"{k}=All:{v['blk_err_px']}px/Mov:{v['mov_blk_err_px']}px(g={v['mean_gate']},rec={v['gate_recall']})"
                    for k, v in ev.items()
                )
            )
            if cem:
                print("   CEM -> " + " | ".join(f"{k}={v['success_rate']*100:.1f}%({v['final_dist_px']}px)" for k, v in cem.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
