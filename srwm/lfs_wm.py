"""Local-Frame Sparse-Interaction World Model (LFS-WM) for AnyGoal PushT (G1c).

Implements:
- 3-frame macro-step kinematics (frame_skip=5, history_length=3)
- SE(2) local-frame edge projections (T-block body frame & agent-block contact frame)
- Exact Pymunk T-block 2-box Swept-Trajectory SDF Clearance along autonomous path
- Shared Newtonian Reciprocal Contact Gate (g_ab_agt == g_ab_blk) -> exact zero free-space leakage
- 4 static arena wall boundary tokens ([5, 506]^2, r=2 -> inner wall at 7, 504)
- Residual-Anchored Anti-Collapse Gate Loss
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

POS_SCALE = 512.0
VEL_SCALE = 300.0
WALL_MIN = 7.0
WALL_MAX = 504.0
WALL_CLEAR_NORM = 64.0 / POS_SCALE  # 0.125 normalized (~64 px)
SWEPT_CUTOFF_NORM = 8.0 / POS_SCALE  # Physical outer swept contact cutoff (8 px surface clearance)


def rot2d(vec: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate 2D vector (..., 2) by +theta (inverse=False) or -theta (inverse=True)."""
    vx, vy = vec[..., 0:1], vec[..., 1:2]
    if inverse:
        rx = cos_t * vx + sin_t * vy
        ry = -sin_t * vx + cos_t * vy
    else:
        rx = cos_t * vx - sin_t * vy
        ry = sin_t * vx + cos_t * vy
    return torch.cat([rx, ry], dim=-1)


def rot2d_seq(seq: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate a sequence of 2D vectors (B, K, 2) using (B, 1) cos/sin."""
    return rot2d(seq, cos_t.unsqueeze(1), sin_t.unsqueeze(1), inverse=inverse)


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [0, 2*pi) matching PushT state[:, 4]."""
    return torch.remainder(theta, 2.0 * math.pi)


def angle_diff_rad(pred_theta: torch.Tensor, gt_theta: torch.Tensor) -> torch.Tensor:
    """Shortest signed angular difference in [-pi, pi]."""
    return torch.atan2(torch.sin(pred_theta - gt_theta), torch.cos(pred_theta - gt_theta))


def t_block_sdf_norm(p_rel_body_norm: torch.Tensor) -> torch.Tensor:
    """Exact surface-to-surface signed clearance (in normalized /512 units) from agent circle
    (radius=15px) to the Pymunk T-block (head bar [-60,60]x[0,30], stem [-15,15]x[30,120]).
    """
    qx = p_rel_body_norm[..., 0:1] * POS_SCALE
    qy = p_rel_body_norm[..., 1:2] * POS_SCALE

    # Box 1 (Horizontal Head Bar): center (0, 15), half-extents (60, 15)
    dx1 = torch.abs(qx) - 60.0
    dy1 = torch.abs(qy - 15.0) - 15.0
    d1_out = torch.hypot(torch.relu(dx1), torch.relu(dy1))
    d1_in = torch.clamp(torch.maximum(dx1, dy1), max=0.0)
    d1 = d1_out + d1_in

    # Box 2 (Vertical Stem): center (0, 75), half-extents (15, 45)
    dx2 = torch.abs(qx) - 15.0
    dy2 = torch.abs(qy - 75.0) - 45.0
    d2_out = torch.hypot(torch.relu(dx2), torch.relu(dy2))
    d2_in = torch.clamp(torch.maximum(dx2, dy2), max=0.0)
    d2 = d2_out + d2_in

    clearance_px = torch.minimum(d1, d2) - 15.0
    return clearance_px / POS_SCALE


def ste_binary_gate(g: torch.Tensor, use_ste: bool, thresh: float = 0.5) -> torch.Tensor:
    """Straight-Through Estimator binary gate for train/eval consistency."""
    if not use_ste:
        return g
    hard = (g >= thresh).to(g.dtype)
    return hard + g - g.detach()


def wall_boundary_features(
    pos_norm: torch.Tensor,
    vel_norm: torch.Tensor,
    act_flat: torch.Tensor,
    cos_ref: torch.Tensor,
    sin_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 4 static arena wall edge features in the receiver's local frame."""
    bsz = pos_norm.shape[0]
    device, dtype = pos_norm.device, pos_norm.dtype
    w_min = WALL_MIN / POS_SCALE
    w_max = WALL_MAX / POS_SCALE

    d_left = pos_norm[:, 0:1] - w_min
    d_right = w_max - pos_norm[:, 0:1]
    d_bot = pos_norm[:, 1:2] - w_min
    d_top = w_max - pos_norm[:, 1:2]
    dists = torch.stack([d_left, d_right, d_bot, d_top], dim=1)  # (B, 4, 1)

    normals_world = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        device=device,
        dtype=dtype,
    ).unsqueeze(0).expand(bsz, 4, 2)

    c4 = cos_ref.unsqueeze(1)
    s4 = sin_ref.unsqueeze(1)
    n_loc = rot2d(normals_world, c4, s4, inverse=True)
    v_loc = rot2d(vel_norm, cos_ref, sin_ref, inverse=True).unsqueeze(1).expand(bsz, 4, 2)
    act_seq = act_flat.view(bsz, -1, 2)
    act_loc = rot2d_seq(act_seq, cos_ref, sin_ref, inverse=True).reshape(bsz, 1, -1).expand(bsz, 4, -1)

    active_mask = (dists < WALL_CLEAR_NORM).to(dtype)
    feat = torch.cat([n_loc, dists, v_loc, act_loc], dim=-1)
    return feat, active_mask


class LFSWorldModel(nn.Module):
    """Unified 2x2 ablation world model over 3-frame macro-step PushT history."""

    def __init__(
        self,
        d_model: int = 128,
        history_length: int = 3,
        frame_skip: int = 5,
        use_sparse_gate: bool = True,
        use_body_frame: bool = True,
        pos_scale: float = POS_SCALE,
        vel_scale: float = VEL_SCALE,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.history_length = int(history_length)
        self.frame_skip = int(frame_skip)
        self.use_sparse_gate = bool(use_sparse_gate)
        self.use_body_frame = bool(use_body_frame)
        self.pos_scale = float(pos_scale)
        self.vel_scale = float(vel_scale)
        act_block_dim = 2 * self.frame_skip  # 10

        if not self.use_body_frame and not self.use_sparse_gate:
            in_dim = self.history_length * (8 + act_block_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 7),
            )
            return

        if self.use_body_frame:
            self.agent_auto_mlp = nn.Sequential(
                nn.Linear(6 + 2 * act_block_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 4),
            )
            self.block_damping_logit = nn.Parameter(torch.tensor(-2.0))

            inter_blk_dim = 16 + act_block_dim
            self.blk_inter_mlp = nn.Sequential(
                nn.Linear(inter_blk_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 3),
            )
            inter_agt_dim = 14 + act_block_dim
            self.agt_inter_mlp = nn.Sequential(
                nn.Linear(inter_agt_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 4),
            )
            wall_dim = 5 + act_block_dim
            self.agt_wall_mlp = nn.Sequential(
                nn.Linear(wall_dim, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 4),
            )
            self.blk_wall_mlp = nn.Sequential(
                nn.Linear(wall_dim, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 3),
            )

            if self.use_sparse_gate:
                self.gate_ab = nn.Sequential(
                    nn.Linear(9, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                    nn.Linear(32, 1),
                )
                nn.init.constant_(self.gate_ab[-1].bias, 0.5)
        else:
            self.agent_auto_mlp = nn.Sequential(
                nn.Linear(6 + 2 * act_block_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 4),
            )
            self.block_damping_logit = nn.Parameter(torch.tensor(-2.0))
            inter_dim = 10 + act_block_dim
            self.blk_inter_mlp = nn.Sequential(
                nn.Linear(inter_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 3),
            )
            self.agt_inter_mlp = nn.Sequential(
                nn.Linear(inter_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 4),
            )
            self.gate_ab = nn.Sequential(nn.Linear(inter_dim, 64), nn.SiLU(), nn.Linear(64, 1))
            nn.init.constant_(self.gate_ab[-1].bias, 0.5)

    def forward_step(
        self,
        s_hist: torch.Tensor,
        a_hist: torch.Tensor,
        use_ste: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict next macro-step state s_{t+frame_skip}."""
        bsz = s_hist.shape[0]
        s_cur = s_hist[:, -1]
        s_prev1 = s_hist[:, -2]
        s_prev2 = s_hist[:, -3] if s_hist.shape[1] >= 3 else s_prev1

        p_a = s_cur[:, 0:2] / self.pos_scale
        p_b = s_cur[:, 2:4] / self.pos_scale
        th = s_cur[:, 4:5]
        v_a = s_cur[:, 5:7] / self.vel_scale
        cos_t, sin_t = torch.cos(th), torch.sin(th)

        a_flat = a_hist.reshape(bsz, self.history_length, -1)
        a_cur = a_flat[:, -1]
        a_prev = a_flat[:, -2]

        if not self.use_body_frame and not self.use_sparse_gate:
            s_norm = torch.cat(
                [
                    s_hist[:, :, 0:2] / self.pos_scale,
                    s_hist[:, :, 2:4] / self.pos_scale,
                    torch.cos(s_hist[:, :, 4:5]),
                    torch.sin(s_hist[:, :, 4:5]),
                    s_hist[:, :, 5:7] / self.vel_scale,
                    a_flat,
                ],
                dim=-1,
            ).reshape(bsz, -1)
            d = self.net(s_norm)
            next_pa = torch.clamp(s_cur[:, 0:2] + d[:, 0:2] * self.pos_scale, 0.0, self.pos_scale)
            next_pb = torch.clamp(s_cur[:, 2:4] + d[:, 2:4] * self.pos_scale, 0.0, self.pos_scale)
            next_th = wrap_angle(s_cur[:, 4:5] + d[:, 4:5])
            next_va = torch.clamp(s_cur[:, 5:7] + d[:, 5:7] * self.vel_scale, -self.vel_scale, self.vel_scale)
            next_s = torch.cat([next_pa, next_pb, next_th, next_va], dim=-1)
            ones = torch.ones(bsz, 1, device=s_cur.device, dtype=s_cur.dtype)
            return next_s, {
                "s_auto": next_s.detach(),
                "g_ab_raw": ones,
                "g_ab_blk": ones,
                "d_swept": torch.zeros_like(ones),
                "mean_gate": torch.tensor(1.0, device=s_cur.device),
            }

        dp_a_h1 = (s_cur[:, 0:2] - s_prev1[:, 0:2]) / self.pos_scale
        dp_a_h2 = (s_prev1[:, 0:2] - s_prev2[:, 0:2]) / self.pos_scale
        dp_b_h1 = (s_cur[:, 2:4] - s_prev1[:, 2:4]) / self.pos_scale
        dth_h1 = angle_diff_rad(s_cur[:, 4:5], s_prev1[:, 4:5])

        # 1. Autonomous Free-Space Streams
        if self.use_body_frame:
            d_auto_a = self.agent_auto_mlp(torch.cat([v_a, dp_a_h1, dp_a_h2, a_cur, a_prev], dim=-1))
        else:
            d_auto_a = self.agent_auto_mlp(torch.cat([p_a, v_a, dp_a_h1, a_cur, a_prev], dim=-1))
        dp_a_auto, dv_a_auto = d_auto_a[:, 0:2], d_auto_a[:, 2:4]

        gamma = 0.5 * torch.sigmoid(self.block_damping_logit)
        dp_b_auto = gamma * dp_b_h1
        dth_auto = gamma * dth_h1

        s_auto = torch.cat(
            [
                torch.clamp(s_cur[:, 0:2] + dp_a_auto * self.pos_scale, 0.0, self.pos_scale),
                torch.clamp(s_cur[:, 2:4] + dp_b_auto * self.pos_scale, 0.0, self.pos_scale),
                wrap_angle(s_cur[:, 4:5] + dth_auto),
                torch.clamp(s_cur[:, 5:7] + dv_a_auto * self.vel_scale, -self.vel_scale, self.vel_scale),
            ],
            dim=-1,
        )

        # 2. Interaction Streams
        if self.use_body_frame:
            dp_a_auto_det = dp_a_auto.detach()
            dp_b_auto_det = dp_b_auto.detach()
            p_rel_0 = rot2d(p_a - p_b, cos_t, sin_t, inverse=True)
            p_rel_mid = rot2d(
                (p_a + 0.5 * dp_a_auto_det) - (p_b + 0.5 * dp_b_auto_det), cos_t, sin_t, inverse=True
            )
            p_rel_1 = rot2d((p_a + dp_a_auto_det) - (p_b + dp_b_auto_det), cos_t, sin_t, inverse=True)

            d_0 = t_block_sdf_norm(p_rel_0)
            d_mid = t_block_sdf_norm(p_rel_mid)
            d_1 = t_block_sdf_norm(p_rel_1)
            d_swept = torch.minimum(torch.minimum(d_0, d_mid), d_1)

            dist_0 = torch.norm(p_rel_0, dim=-1, keepdim=True) + 1e-6
            dist_1 = torch.norm(p_rel_1, dim=-1, keepdim=True) + 1e-6
            v_a_b = rot2d(v_a, cos_t, sin_t, inverse=True)
            dp_a_b = rot2d(dp_a_h1, cos_t, sin_t, inverse=True)
            dp_b_b = rot2d(dp_b_h1, cos_t, sin_t, inverse=True)
            a_cur_b = rot2d_seq(a_cur.view(bsz, self.frame_skip, 2), cos_t, sin_t, inverse=True).reshape(bsz, -1)

            feat_ab_blk = torch.cat(
                [p_rel_0, p_rel_1, d_0, d_mid, d_1, d_swept, dist_0, v_a_b, dp_a_b, dp_b_b, dth_h1, a_cur_b],
                dim=-1,
            )
            d_blk_ab_loc = self.blk_inter_mlp(feat_ab_blk)
            dp_b_ab = rot2d(d_blk_ab_loc[:, 0:2], cos_t, sin_t, inverse=False)
            dth_ab = d_blk_ab_loc[:, 2:3]

            dp_ab_world = p_a - p_b
            normal_ab = dp_ab_world / dist_0
            cos_ij, sin_ij = normal_ab[:, 0:1], normal_ab[:, 1:2]
            v_a_ij = rot2d(v_a, cos_ij, sin_ij, inverse=True)
            dp_a_ij = rot2d(dp_a_h1, cos_ij, sin_ij, inverse=True)
            dp_b_ij = rot2d(dp_b_h1, cos_ij, sin_ij, inverse=True)
            rel_orient = rot2d(torch.cat([cos_t, sin_t], dim=-1), cos_ij, sin_ij, inverse=True)
            a_cur_ij = rot2d_seq(a_cur.view(bsz, self.frame_skip, 2), cos_ij, sin_ij, inverse=True).reshape(bsz, -1)
            feat_ab_agt = torch.cat(
                [dist_0, d_0, d_mid, d_1, d_swept, v_a_ij, dp_a_ij, dp_b_ij, dth_h1, rel_orient, a_cur_ij],
                dim=-1,
            )
            d_agt_ab_loc = self.agt_inter_mlp(feat_ab_agt)
            dp_a_ab = rot2d(d_agt_ab_loc[:, 0:2], cos_ij, sin_ij, inverse=False)
            dv_a_ab = rot2d(d_agt_ab_loc[:, 2:4], cos_ij, sin_ij, inverse=False)

            feat_wb, mask_wb = wall_boundary_features(p_b, dp_b_h1, torch.zeros_like(a_cur), cos_t, sin_t)
            d_blk_w_loc = self.blk_wall_mlp(feat_wb)
            ones_cos = torch.ones_like(cos_t)
            zeros_sin = torch.zeros_like(sin_t)
            feat_wa, mask_wa = wall_boundary_features(p_a, v_a, a_cur, ones_cos, zeros_sin)
            d_agt_w_loc = self.agt_wall_mlp(feat_wa)

            if self.use_sparse_gate:
                v_rel_b = dp_a_auto_det - dp_b_b
                v_rel_norm = torch.norm(v_rel_b, dim=-1, keepdim=True)
                approach_speed = d_0 - d_1
                dp_b_norm = torch.norm(dp_b_h1, dim=-1, keepdim=True)
                gate_in_iso = torch.cat(
                    [d_0, d_mid, d_1, d_swept, dist_0, dist_1, v_rel_norm, approach_speed, dp_b_norm],
                    dim=-1,
                )
                outer_mask = (d_swept < SWEPT_CUTOFF_NORM).to(s_cur.dtype)
                g_ab_logit = self.gate_ab(gate_in_iso)
                g_ab_raw = torch.sigmoid(g_ab_logit) * outer_mask
                g_ab_det = g_ab_raw.detach()
                g_ab = ((g_ab_det >= 0.5).to(s_cur.dtype) if use_ste else g_ab_det) * outer_mask
                mean_gate = g_ab.mean()
            else:
                g_ab_logit = torch.ones(bsz, 1, device=s_cur.device, dtype=s_cur.dtype)
                g_ab_raw = g_ab_logit
                g_ab = g_ab_raw
                mean_gate = torch.tensor(1.0, device=s_cur.device)

            dp_b_w_loc = (mask_wb * d_blk_w_loc[:, :, 0:2]).sum(dim=1)
            dth_w = (mask_wb * d_blk_w_loc[:, :, 2:3]).sum(dim=1)
            dp_b_w = rot2d(dp_b_w_loc, cos_t, sin_t, inverse=False)

            dp_a_w = (mask_wa * d_agt_w_loc[:, :, 0:2]).sum(dim=1)
            dv_a_w = (mask_wa * d_agt_w_loc[:, :, 2:4]).sum(dim=1)

            dp_a = (dp_a_auto + g_ab * dp_a_ab + dp_a_w) * self.pos_scale
            dv_a = (dv_a_auto + g_ab * dv_a_ab + dv_a_w) * self.vel_scale
            dp_b = (dp_b_auto + g_ab * dp_b_ab + dp_b_w) * self.pos_scale
            dth = dth_auto + g_ab * dth_ab + dth_w
        else:
            # M1: Sparse-Global
            feat_glob = torch.cat([p_a, p_b, cos_t, sin_t, v_a, dp_b_h1, a_cur], dim=-1)
            d_blk = self.blk_inter_mlp(feat_glob)
            d_agt = self.agt_inter_mlp(feat_glob)
            g_ab_logit = self.gate_ab(feat_glob)
            g_ab_raw = torch.sigmoid(g_ab_logit)
            g_ab_det = g_ab_raw.detach()
            g_ab = (g_ab_det >= 0.5).to(s_cur.dtype) if use_ste else g_ab_det
            d_swept = torch.zeros_like(g_ab_raw)
            mean_gate = g_ab.mean()

            dp_a = (dp_a_auto + g_ab * d_agt[:, 0:2]) * self.pos_scale
            dv_a = (dv_a_auto + g_ab * d_agt[:, 2:4]) * self.vel_scale
            dp_b = (dp_b_auto + g_ab * d_blk[:, 0:2]) * self.pos_scale
            dth = dth_auto + g_ab * d_blk[:, 2:3]

        next_pa = torch.clamp(s_cur[:, 0:2] + dp_a, 0.0, self.pos_scale)
        next_pb = torch.clamp(s_cur[:, 2:4] + dp_b, 0.0, self.pos_scale)
        next_th = wrap_angle(s_cur[:, 4:5] + dth)
        next_va = torch.clamp(s_cur[:, 5:7] + dv_a, -self.vel_scale, self.vel_scale)
        next_s = torch.cat([next_pa, next_pb, next_th, next_va], dim=-1)

        return next_s, {
            "s_auto": s_auto,
            "g_ab_logit": g_ab_logit,
            "g_ab_raw": g_ab_raw,
            "g_ab_blk": g_ab_raw,
            "d_swept": d_swept,
            "mean_gate": mean_gate,
        }


def pusht_state_error(
    pred: torch.Tensor, gt: torch.Tensor, pos_scale: float = POS_SCALE, vel_scale: float = VEL_SCALE
) -> torch.Tensor:
    """Per-sample normalized state error (B,) matching PushT physical scales."""
    l_pa = ((pred[:, 0:2] - gt[:, 0:2]) / pos_scale).square().mean(dim=-1)
    l_pb = ((pred[:, 2:4] - gt[:, 2:4]) / pos_scale).square().mean(dim=-1)
    l_th = (angle_diff_rad(pred[:, 4:5], gt[:, 4:5]).square().squeeze(-1)) / (math.pi**2)
    l_va = ((pred[:, 5:7] - gt[:, 5:7]) / vel_scale).square().mean(dim=-1)
    return l_pa + 2.0 * l_pb + 1.5 * l_th + 0.5 * l_va


def residual_anchored_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    aux: dict[str, torch.Tensor],
    use_sparse_gate: bool,
    pos_scale: float = POS_SCALE,
    vel_scale: float = VEL_SCALE,
) -> torch.Tensor:
    """Anti-collapse Residual-Anchored Sparse Gating Loss with Newtonian Reciprocal Gate."""
    per_sample_err = pusht_state_error(pred, gt, pos_scale, vel_scale)
    if not use_sparse_gate:
        return per_sample_err.mean()

    s_auto = aux["s_auto"].detach()
    dp_b_res_px = torch.norm(gt[:, 2:4] - s_auto[:, 2:4], dim=-1, keepdim=True)
    dth_res_rad = torch.abs(angle_diff_rad(gt[:, 4:5], s_auto[:, 4:5]))
    y_blk = ((dp_b_res_px > 1.5) | (dth_res_rad > 0.01)).to(gt.dtype)

    # 1. Prediction loss (mild contact upweighting so moving windows are never drowned out)
    y_flat = y_blk.squeeze(-1)
    w_sample = 1.0 + 1.5 * y_flat
    l_pred = (w_sample * per_sample_err).mean()

    # 2. Pure autonomous agent supervision on strictly non-contact steps (y_blk == 0)
    s_auto_live = aux["s_auto"]
    mask_free = 1.0 - y_flat
    err_auto_agt = (
        ((s_auto_live[:, 0:2] - gt[:, 0:2]) / pos_scale).square().mean(dim=-1)
        + 0.5 * ((s_auto_live[:, 5:7] - gt[:, 5:7]) / vel_scale).square().mean(dim=-1)
    )
    l_auto = (mask_free * err_auto_agt).sum() / mask_free.sum().clamp_min(1.0)

    # 3. Reciprocal gate anchor BCE supervised purely by true block motion y_blk (detached from l_pred)
    l_anchor = F.binary_cross_entropy_with_logits(aux["g_ab_logit"], y_blk)

    return l_pred + 0.30 * l_auto + 0.20 * l_anchor
