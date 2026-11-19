"""G0 Planar Push-Block SE(2) physics, dataset generator, and CEM planner."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from srwm.models import FactoredWorldModel

DEFAULT_G0_CFG = {
    "dt": 0.1,
    "r_agent": 0.12,
    "r_block": 0.15,
    "damping_agent": 0.75,
    "damping_block": 0.65,
    "act_gain": 0.45,
    "k_rep": 3.2,
    "k_push": 0.75,
    "ep_len": 25,
    "gate_hard_thresh": 0.05,
    "cem_horizon": 10,
    "cem_pop": 128,
    "cem_elites": 16,
    "cem_iters": 3,
    "cem_episodes": 32,
}


def physics_step(states: np.ndarray, actions: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """True SE(2)-equivariant multi-entity planar contact dynamics."""
    dt = cfg["dt"]
    bsz, n_ent, _ = states.shape
    pos = states[:, :, 0:2].copy()
    vel = states[:, :, 2:4].copy()

    radii = np.full(n_ent, cfg["r_block"], dtype=np.float32)
    radii[0] = cfg["r_agent"]
    r_sum = radii[:, None] + radii[None, :]

    vel_new = np.zeros_like(vel)
    vel_new[:, 0] = cfg["damping_agent"] * vel[:, 0] + cfg["act_gain"] * actions
    vel_new[:, 1:] = cfg["damping_block"] * vel[:, 1:]

    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + 1e-8
    eye_mask = ~np.eye(n_ent, dtype=bool)[None, :, :]
    contact = (dist < r_sum[None, :, :]) & eye_mask
    pen = np.maximum(0.0, r_sum[None, :, :] - dist) * contact
    normal = diff / dist[:, :, :, None]

    v_diff = vel_new[:, None, :, :] - vel_new[:, :, None, :]
    v_app = np.maximum(0.0, np.sum(v_diff * normal, axis=-1)) * contact

    act_full = np.zeros((bsz, n_ent, 2), dtype=np.float32)
    act_full[:, 0, :] = actions
    a_diff = act_full[:, None, :, :] - act_full[:, :, None, :]
    a_push = np.maximum(0.0, np.sum(a_diff * normal, axis=-1)) * contact

    impulse_mag = cfg["k_rep"] * pen + 0.55 * v_app + cfg["k_push"] * a_push
    dv_int = np.sum(impulse_mag[:, :, :, None] * normal, axis=2)

    vel_final = np.clip(vel_new + dv_int, -3.0, 3.0)
    pos_final = np.clip(pos + dt * vel_final, -1.5, 1.5)
    next_states = np.concatenate([pos_final, vel_final], axis=-1).astype(np.float32)
    return next_states, contact


def generate_dataset(mode: str, n_episodes: int, ep_len: int, seed: int, cfg: dict) -> dict[str, np.ndarray]:
    """Generate id_center (upward bias), avoid, anti_goal (downward), or multi_3 trajectories."""
    rng = np.random.default_rng(seed)
    n_ent = 4 if mode == "multi_3" else 2
    states = np.zeros((n_episodes, ep_len + 1, n_ent, 4), dtype=np.float32)
    actions = np.zeros((n_episodes, ep_len, 2), dtype=np.float32)
    contacts = np.zeros((n_episodes, ep_len), dtype=bool)

    if mode == "id_center":
        p_blk = np.stack(
            [rng.uniform(-0.65, 0.65, size=n_episodes), rng.uniform(-0.60, 0.35, size=n_episodes)],
            axis=-1,
        ).astype(np.float32)
        goals = np.tile(np.array([[0.0, 0.75]], dtype=np.float32), (n_episodes, 1))
        push_dir = goals - p_blk
        push_dir /= np.linalg.norm(push_dir, axis=-1, keepdims=True) + 1e-6
        offset_ang = rng.uniform(-0.6, 0.6, size=n_episodes)
        c, s = np.cos(offset_ang), np.sin(offset_ang)
        rot_dir = np.stack([c * push_dir[:, 0] - s * push_dir[:, 1], s * push_dir[:, 0] + c * push_dir[:, 1]], axis=-1)
        p_agt = p_blk - rot_dir * (
            cfg["r_agent"] + cfg["r_block"] + rng.uniform(0.01, 0.14, size=(n_episodes, 1))
        ).astype(np.float32)
        states[:, 0, 0, 0:2] = p_agt
        states[:, 0, 1, 0:2] = p_blk
    elif mode == "avoid":
        angles = rng.uniform(0, 2 * np.pi, size=n_episodes)
        unit_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
        ortho = np.stack([-unit_dirs[:, 1], unit_dirs[:, 0]], axis=-1)
        p_blk = rng.uniform(-0.45, 0.45, size=(n_episodes, 2)).astype(np.float32)
        p_agt = p_blk + ortho * rng.uniform(0.45, 0.68, size=(n_episodes, 1)).astype(np.float32) - unit_dirs * 0.4
        goals = p_blk + ortho * 0.55 + unit_dirs * 0.5
        states[:, 0, 0, 0:2] = np.clip(p_agt, -1.0, 1.0)
        states[:, 0, 1, 0:2] = p_blk
    elif mode == "anti_goal":
        p_blk = np.stack(
            [rng.uniform(-0.40, 0.40, size=n_episodes), rng.uniform(-0.10, 0.35, size=n_episodes)],
            axis=-1,
        ).astype(np.float32)
        goals = np.stack(
            [rng.choice([-0.65, 0.65], size=n_episodes), rng.uniform(-0.85, -0.60, size=n_episodes)],
            axis=-1,
        ).astype(np.float32)
        push_dir = goals - p_blk
        push_dir /= np.linalg.norm(push_dir, axis=-1, keepdims=True) + 1e-6
        p_agt = p_blk - push_dir * (
            cfg["r_agent"] + cfg["r_block"] + rng.uniform(0.01, 0.08, size=(n_episodes, 1))
        ).astype(np.float32)
        states[:, 0, 0, 0:2] = p_agt
        states[:, 0, 1, 0:2] = p_blk
    else:
        angles = rng.uniform(-np.pi, 0, size=n_episodes)
        unit_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
        ortho = np.stack([-unit_dirs[:, 1], unit_dirs[:, 0]], axis=-1)
        base_p = rng.uniform(-0.3, 0.3, size=(n_episodes, 2)).astype(np.float32)
        states[:, 0, 0, 0:2] = base_p - unit_dirs * 0.36
        states[:, 0, 1, 0:2] = base_p
        states[:, 0, 2, 0:2] = base_p + unit_dirs * 0.24 + ortho * 0.14
        states[:, 0, 3, 0:2] = base_p + unit_dirs * 0.24 - ortho * 0.14
        goals = base_p + unit_dirs * 0.85

    cur_s = states[:, 0].copy()
    for t in range(ep_len):
        if mode == "avoid":
            des = goals - cur_s[:, 0, 0:2]
            rel = cur_s[:, 0, 0:2] - cur_s[:, 1, 0:2]
            d = np.linalg.norm(rel, axis=-1, keepdims=True) + 1e-6
            rep = (rel / d) * np.maximum(0.0, 0.55 - d) * 3.0
            raw_a = 2.2 * des + rep + rng.normal(0, 0.15, size=des.shape)
        elif mode in ("id_center", "anti_goal"):
            blk_to_goal = goals - cur_s[:, 1, 0:2]
            dist_bg = np.linalg.norm(blk_to_goal, axis=-1, keepdims=True) + 1e-6
            push_dir = blk_to_goal / dist_bg
            push_target = cur_s[:, 1, 0:2] - push_dir * (cfg["r_agent"] + cfg["r_block"] - 0.03)
            err = push_target - cur_s[:, 0, 0:2]
            raw_a = 4.0 * err + 1.2 * push_dir + rng.normal(0, 0.22, size=err.shape)
        else:
            des = goals - cur_s[:, 0, 0:2]
            raw_a = 2.5 * des + rng.normal(0, 0.20, size=des.shape)

        act = np.clip(raw_a, -1.0, 1.0).astype(np.float32)
        next_s, c_mat = physics_step(cur_s, act, cfg)
        actions[:, t] = act
        states[:, t + 1] = next_s
        contacts[:, t] = np.any(c_mat, axis=(1, 2))
        cur_s = next_s

    return {"states": states, "actions": actions, "contacts": contacts, "goals": goals}


@torch.no_grad()
def evaluate_cem_planning(
    model: FactoredWorldModel,
    mode: str,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> dict:
    model.eval()
    ds = generate_dataset(mode, cfg["cem_episodes"], cfg["ep_len"], seed + 9000, cfg)
    cur_s = ds["states"][:, 0].copy()
    goals = ds["goals"].copy()
    n_ep = cur_s.shape[0]
    h_plan, pop, elites = cfg["cem_horizon"], cfg["cem_pop"], cfg["cem_elites"]
    thresh = cfg["gate_hard_thresh"]

    ever_collided = np.zeros(n_ep, dtype=bool)
    goals_t = torch.from_numpy(goals).to(device)

    for _ in range(cfg["ep_len"]):
        s_t = torch.from_numpy(cur_s).to(device)
        mu = torch.zeros(n_ep, h_plan, 2, device=device)
        std = torch.ones(n_ep, h_plan, 2, device=device) * 0.8

        for _ in range(cfg["cem_iters"]):
            eps = torch.randn(n_ep, pop, h_plan, 2, device=device)
            act_seq = torch.clamp(mu.unsqueeze(1) + std.unsqueeze(1) * eps, -1.0, 1.0)
            z_sim = s_t.unsqueeze(1).expand(n_ep, pop, 2, 4).reshape(n_ep * pop, 2, 4)
            cost = torch.zeros(n_ep * pop, device=device)
            g_rep = goals_t.unsqueeze(1).expand(n_ep, pop, 2).reshape(n_ep * pop, 2)

            for tau in range(h_plan):
                a_tau = act_seq[:, :, tau, :].reshape(n_ep * pop, 2)
                z_sim, _ = model.forward_step(z_sim, a_tau, hard_gate_thresh=thresh)
                if mode == "avoid":
                    dist_goal = torch.norm(z_sim[:, 0, 0:2] - g_rep, dim=-1)
                    dist_blk = torch.norm(z_sim[:, 0, 0:2] - z_sim[:, 1, 0:2], dim=-1)
                    cost = cost + dist_goal + F.relu(0.34 - dist_blk) * 15.0
                else:
                    dist_blk_goal = torch.norm(z_sim[:, 1, 0:2] - g_rep, dim=-1)
                    dist_agt_blk = torch.norm(z_sim[:, 0, 0:2] - z_sim[:, 1, 0:2], dim=-1)
                    cost = cost + 2.5 * dist_blk_goal + 0.4 * dist_agt_blk

            cost = cost.view(n_ep, pop)
            top_idx = torch.topk(-cost, k=elites, dim=-1).indices
            gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(n_ep, elites, h_plan, 2)
            elite_acts = torch.gather(act_seq, 1, gather_idx)
            mu = elite_acts.mean(dim=1)
            std = elite_acts.std(dim=1) + 0.05

        best_a = mu[:, 0, :].cpu().numpy()
        cur_s, c_mat = physics_step(cur_s, best_a, cfg)
        ever_collided |= np.any(c_mat, axis=(1, 2))

    if mode == "avoid":
        final_dist = np.linalg.norm(cur_s[:, 0, 0:2] - goals, axis=-1)
        success = (final_dist < 0.25) & (~ever_collided)
    else:
        final_dist = np.linalg.norm(cur_s[:, 1, 0:2] - goals, axis=-1)
        success = final_dist < 0.28

    return {"final_dist": round(float(final_dist.mean()), 4), "success_rate": round(float(success.mean()), 4)}
