"""Rigid-body SE(2) body-frame world model for AnyGoal PushT (M0-M3 2x2 ablation)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [0, 2*pi) matching PushT state[:, 4]."""
    return torch.remainder(theta, 2.0 * math.pi)


def angle_diff_rad(pred_theta: torch.Tensor, gt_theta: torch.Tensor) -> torch.Tensor:
    """Shortest signed angular difference in [-pi, pi]."""
    return torch.atan2(torch.sin(pred_theta - gt_theta), torch.cos(pred_theta - gt_theta))


class PushTWorldModel(nn.Module):
    """
    2x2 ablation for real PushT 7D rigid-body state [xa, ya, xb, yb, theta_b, vxa, vya]:
    - M0 (sparse=False, body=False): Holistic Global MLP
    - M1 (sparse=True,  body=False): Sparse-Global (SPARTAN-style global gate + global interaction)
    - M2 (sparse=False, body=True):  Dense-BodyFrame (SE(2) T-block body frame, no sparse gate)
    - M3 (sparse=True,  body=True):  SR-WM-Rigid (Ours: SE(2) T-block body frame + sparse contact gate)
    """

    def __init__(
        self,
        d_model: int = 128,
        use_sparse_gate: bool = True,
        use_body_frame: bool = True,
        pos_scale: float = 512.0,
        vel_scale: float = 300.0,
    ) -> None:
        super().__init__()
        self.use_sparse_gate = use_sparse_gate
        self.use_body_frame = use_body_frame
        self.pos_scale = float(pos_scale)
        self.vel_scale = float(vel_scale)

        if not use_body_frame and not use_sparse_gate:
            self.net = nn.Sequential(
                nn.Linear(10, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 7),
            )
        else:
            agt_in = 4 if use_body_frame else 6
            self.agent_mlp = nn.Sequential(
                nn.Linear(agt_in, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 4),
            )
            int_in = 7 if use_body_frame else 10
            self.inter_mlp = nn.Sequential(
                nn.Linear(int_in, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 7),
            )
            if use_sparse_gate:
                gate_in = 7 if use_body_frame else 8
                self.gate_mlp = nn.Sequential(
                    nn.Linear(gate_in, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1),
                )
                nn.init.constant_(self.gate_mlp[-1].bias, 1.0)

    def forward_step(
        self, s: torch.Tensor, a: torch.Tensor, hard_gate_thresh: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """s: (B, 7) raw physical units; a: (B, 2) in [-1, 1]."""
        p_a = s[:, 0:2] / self.pos_scale
        p_b = s[:, 2:4] / self.pos_scale
        th = s[:, 4:5]
        v_a = s[:, 5:7] / self.vel_scale
        cos_t, sin_t = torch.cos(th), torch.sin(th)

        if not self.use_body_frame and not self.use_sparse_gate:
            x = torch.cat([p_a, p_b, cos_t, sin_t, v_a, a], dim=-1)
            d = self.net(x)
            dp_a = d[:, 0:2] * self.pos_scale
            dp_b = d[:, 2:4] * self.pos_scale
            dth = d[:, 4:5]
            dv_a = d[:, 5:7] * self.vel_scale
            g_mean = torch.tensor(1.0, device=s.device)
        else:
            agt_in = torch.cat([v_a, a], dim=-1) if self.use_body_frame else torch.cat([p_a, v_a, a], dim=-1)
            d_auto = self.agent_mlp(agt_in)
            dp_a_auto, dv_a_auto = d_auto[:, 0:2], d_auto[:, 2:4]

            if self.use_body_frame:
                p_rel_body = rot2d(p_a - p_b, cos_t, sin_t, inverse=True)
                dist = torch.norm(p_rel_body, dim=-1, keepdim=True)
                v_a_body = rot2d(v_a, cos_t, sin_t, inverse=True)
                a_body = rot2d(a, cos_t, sin_t, inverse=True)
                feat = torch.cat([p_rel_body, dist, v_a_body, a_body], dim=-1)
                d_int_body = self.inter_mlp(feat)
                dp_a_int = rot2d(d_int_body[:, 0:2], cos_t, sin_t, inverse=False)
                dp_b_int = rot2d(d_int_body[:, 2:4], cos_t, sin_t, inverse=False)
                dth_int = d_int_body[:, 4:5]
                dv_a_int = rot2d(d_int_body[:, 5:7], cos_t, sin_t, inverse=False)
                gate_in = feat
            else:
                feat = torch.cat([p_a, p_b, cos_t, sin_t, v_a, a], dim=-1)
                d_int = self.inter_mlp(feat)
                dp_a_int, dp_b_int, dth_int, dv_a_int = d_int[:, 0:2], d_int[:, 2:4], d_int[:, 4:5], d_int[:, 5:7]
                gate_in = torch.cat([p_a, p_b, cos_t, sin_t, v_a], dim=-1)

            if self.use_sparse_gate:
                g = torch.sigmoid(self.gate_mlp(gate_in))
                if hard_gate_thresh > 0.0 and not self.training:
                    g = torch.where(g >= hard_gate_thresh, g, torch.zeros_like(g))
                g_mean = g.mean()
            else:
                g = torch.ones_like(th)
                g_mean = torch.tensor(1.0, device=s.device)

            dp_a = (dp_a_auto + g * dp_a_int) * self.pos_scale
            dp_b = (g * dp_b_int) * self.pos_scale
            dth = g * dth_int
            dv_a = (dv_a_auto + g * dv_a_int) * self.vel_scale

        next_pa = torch.clamp(s[:, 0:2] + dp_a, 0.0, self.pos_scale)
        next_pb = torch.clamp(s[:, 2:4] + dp_b, 0.0, self.pos_scale)
        next_th = wrap_angle(s[:, 4:5] + dth)
        next_va = torch.clamp(s[:, 5:7] + dv_a, -self.vel_scale, self.vel_scale)
        return torch.cat([next_pa, next_pb, next_th, next_va], dim=-1), g_mean


def pusht_state_loss(
    pred: torch.Tensor, gt: torch.Tensor, pos_scale: float = 512.0, vel_scale: float = 300.0
) -> torch.Tensor:
    """Normalized physical loss with wrapped angular distance."""
    l_pa = F.mse_loss(pred[:, 0:2] / pos_scale, gt[:, 0:2] / pos_scale)
    l_pb = F.mse_loss(pred[:, 2:4] / pos_scale, gt[:, 2:4] / pos_scale)
    l_th = torch.mean(angle_diff_rad(pred[:, 4:5], gt[:, 4:5]) ** 2) / (math.pi**2)
    l_va = F.mse_loss(pred[:, 5:7] / vel_scale, gt[:, 5:7] / vel_scale)
    return l_pa + 2.0 * l_pb + 1.5 * l_th + 0.5 * l_va
