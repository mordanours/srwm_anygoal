"""Core SE(2) local-contact-frame & sparse-regime factored world model architectures."""

from __future__ import annotations

import torch
import torch.nn as nn


def to_frame(vec: torch.Tensor, normal: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Project 2D vector (..., 2) onto local orthonormal frame (normal, tangent)."""
    vn = (vec * normal).sum(dim=-1, keepdim=True)
    vt = (vec * tangent).sum(dim=-1, keepdim=True)
    return torch.cat([vn, vt], dim=-1)


def from_frame(local_vec: torch.Tensor, normal: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Rotate 2D vector (..., 2) from local frame back to world coordinates."""
    return local_vec[..., 0:1] * normal + local_vec[..., 1:2] * tangent


class FactoredWorldModel(nn.Module):
    """
    2x2 ablation world model over planar entity states z_i = (px, py, vx, vy):
    - use_sparse_gate=False, use_relational=False -> M0: Holistic Global
    - use_sparse_gate=True,  use_relational=False -> M1: Sparse-Global (SPARTAN-style)
    - use_sparse_gate=False, use_relational=True  -> M2: Dense-Relational (SE(2) frame)
    - use_sparse_gate=True,  use_relational=True  -> M3: SR-WM (Ours)
    """

    def __init__(self, d_model: int = 128, use_sparse_gate: bool = True, use_relational: bool = True) -> None:
        super().__init__()
        self.use_sparse_gate = use_sparse_gate
        self.use_relational = use_relational

        self_in_dim = 5 if use_relational else 7
        self.self_mlp = nn.Sequential(
            nn.Linear(self_in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 4),
        )

        pair_in_dim = 11 if use_relational else 14
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 4),
        )

        if use_sparse_gate:
            gate_in_dim = 5 if use_relational else 8
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            nn.init.constant_(self.gate_mlp[-1].bias, 1.0)

    def forward_step(
        self,
        z: torch.Tensor,
        a_agent: torch.Tensor,
        hard_gate_thresh: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_ent, _ = z.shape
        device = z.device
        pos = z[:, :, 0:2]
        vel = z[:, :, 2:4]

        act_all = torch.zeros(bsz, n_ent, 2, device=device, dtype=z.dtype)
        act_all[:, 0, :] = a_agent
        is_agent = torch.zeros(bsz, n_ent, 1, device=device, dtype=z.dtype)
        is_agent[:, 0, 0] = 1.0

        self_in = (
            torch.cat([vel, act_all, is_agent], dim=-1)
            if self.use_relational
            else torch.cat([pos, vel, act_all, is_agent], dim=-1)
        )
        dz_self = self.self_mlp(self_in)

        p_i = pos.unsqueeze(2).expand(bsz, n_ent, n_ent, 2)
        p_j = pos.unsqueeze(1).expand(bsz, n_ent, n_ent, 2)
        v_i = vel.unsqueeze(2).expand(bsz, n_ent, n_ent, 2)
        v_j = vel.unsqueeze(1).expand(bsz, n_ent, n_ent, 2)
        a_i = act_all.unsqueeze(2).expand(bsz, n_ent, n_ent, 2)
        a_j = act_all.unsqueeze(1).expand(bsz, n_ent, n_ent, 2)
        t_i = is_agent.unsqueeze(2).expand(bsz, n_ent, n_ent, 1)
        t_j = is_agent.unsqueeze(1).expand(bsz, n_ent, n_ent, 1)

        off_diag = (~torch.eye(n_ent, dtype=torch.bool, device=device)).float().view(1, n_ent, n_ent, 1)

        if self.use_relational:
            dp = p_i - p_j
            dist = torch.norm(dp, dim=-1, keepdim=True) + 1e-6
            normal = dp / dist
            tangent = torch.stack([-normal[..., 1], normal[..., 0]], dim=-1)

            v_i_loc = to_frame(v_i, normal, tangent)
            v_j_loc = to_frame(v_j, normal, tangent)
            a_i_loc = to_frame(a_i, normal, tangent)
            a_j_loc = to_frame(a_j, normal, tangent)

            pair_in = torch.cat([dist, v_i_loc, v_j_loc, a_i_loc, a_j_loc, t_i, t_j], dim=-1)
            dz_pair_loc = self.pair_mlp(pair_in)
            dz_pair = torch.cat(
                [
                    from_frame(dz_pair_loc[..., 0:2], normal, tangent),
                    from_frame(dz_pair_loc[..., 2:4], normal, tangent),
                ],
                dim=-1,
            )
            if self.use_sparse_gate:
                gate_in = torch.cat([dist, v_i_loc - v_j_loc, a_i_loc - a_j_loc], dim=-1)
        else:
            pair_in = torch.cat([p_i, p_j, v_i, v_j, a_i, a_j, t_i, t_j], dim=-1)
            dz_pair = self.pair_mlp(pair_in)
            if self.use_sparse_gate:
                gate_in = torch.cat([p_i, p_j, v_i, v_j], dim=-1)

        if self.use_sparse_gate:
            g_ij = torch.sigmoid(self.gate_mlp(gate_in)) * off_diag
            if hard_gate_thresh > 0.0 and not self.training:
                g_ij = torch.where(g_ij >= hard_gate_thresh, g_ij, torch.zeros_like(g_ij))
            gate_l1 = g_ij.sum() / max(1.0, float(bsz * n_ent * (n_ent - 1)))
        else:
            g_ij = off_diag
            gate_l1 = torch.tensor(1.0, device=device)

        dz_int = (g_ij * dz_pair).sum(dim=2)
        return z + dz_self + dz_int, gate_l1
