# src/utils/smooth_arap_loss.py

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn
import igl


class SmoothARAP(nn.Module):
    """
    向量化版 Smooth-ARAP：
    - 不再在 Python 里 for 每个顶点
    - 边的贡献一次性搞成 (E,3,3)，用 index_add_ 聚到 (N,3,3) 上
    - 然后对 (N,3,3) 做 batched SVD
    
    这样复杂度还是 O(E)，但基本都在 GPU 上跑，速度会比原来快一大截。
    """

    def __init__(
        self,
        verts0: torch.Tensor,        # (N,3) float32
        faces: torch.Tensor,         # (F,3) long
        device: torch.device | str,
        lambda_arap: float = 1.0,
        lambda_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        
        v0_np = verts0.detach().cpu().numpy()
        f_np  = faces.detach().cpu().numpy().astype(np.int32)
        self.N: int = v0_np.shape[0]
        self.lambda_arap   = float(lambda_arap)
        self.lambda_smooth = float(lambda_smooth)
        
        # ---------- 构建 cot-Laplacian + 质量 ----------
        L = igl.cotmatrix(v0_np, f_np)  # scipy.sparse (N,N)
        M = igl.massmatrix(v0_np, f_np, igl.MASSMATRIX_TYPE_VORONOI)
        area_v = np.array(M.diagonal()).astype(np.float32)
        area_v[area_v <= 1e-12] = 1e-12
        
        # ---------- Laplacian 邻接边列表 ----------
        L_coo = L.tocoo()
        rows = L_coo.row
        cols = L_coo.col
        vals = L_coo.data
        mask = rows != cols           # 去掉对角线
        rows = rows[mask].astype(np.int64)
        cols = cols[mask].astype(np.int64)
        w_ij = (-vals[mask]).astype(np.float32)   # -L_ij > 0
        w_ij = np.nan_to_num(w_ij, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 原始边向量 e0 = v_j - v_i
        e0 = v0_np[cols] - v0_np[rows]           # (E,3)
        
        # 顶点 valence，用来识别"孤立点"
        valence = np.zeros(self.N, dtype=np.int32)
        for i in rows:
            valence[i] += 1
        
        # 原始 Laplacian 向量 ell0
        ell0 = np.zeros_like(v0_np, dtype=np.float32)
        for e_idx in range(rows.shape[0]):
            i = rows[e_idx]
            j = cols[e_idx]
            wij = w_ij[e_idx]
            if not np.isfinite(wij):
                continue
            ell0[i] += wij * (v0_np[j] - v0_np[i])
        ell0 = np.nan_to_num(ell0, nan=0.0, posinf=0.0, neginf=0.0)
        ell0 = ell0 / area_v[:, None]
        
        dev = torch.device(device)
        
        # 存成 buffer
        self.register_buffer("edge_rows",    torch.from_numpy(rows).long().to(dev))  # (E,)
        self.register_buffer("edge_cols",    torch.from_numpy(cols).long().to(dev))  # (E,)
        self.register_buffer("edge_weights", torch.from_numpy(w_ij).float().to(dev)) # (E,)
        self.register_buffer("e0",           torch.from_numpy(e0).float().to(dev))   # (E,3)
        self.register_buffer("area_v",       torch.from_numpy(area_v).float().to(dev))  # (N,)
        self.register_buffer("ell0",         torch.from_numpy(ell0).float().to(dev))    # (N,3)
        self.register_buffer("valence",      torch.from_numpy(valence).long().to(dev))  # (N,)
        self.register_buffer("I3", torch.eye(3, device=dev))

    def _laplacian_vec(self, verts: torch.Tensor) -> torch.Tensor:
        """
        ell_v = (1/A_v) * Σ_j w_ij (v_j - v_i)
        全向量化版本：按边做 index_add。
        """
        E = self.edge_rows.shape[0]
        v_i = verts[self.edge_rows]      # (E,3)
        v_j = verts[self.edge_cols]      # (E,3)
        diff = v_j - v_i                 # (E,3)
        w = self.edge_weights.view(E, 1) # (E,1)
        contrib = w * diff               # (E,3)
        
        ell = torch.zeros_like(verts)
        ell.index_add_(0, self.edge_rows, contrib)
        ell = ell / self.area_v.view(-1, 1)
        return ell

    def forward(self, verts: torch.Tensor) -> torch.Tensor:
        """
        输入: verts (N,3) 当前 V'
        输出: 标量 loss = λ_arap * E_arap + λ_smooth * E_smooth
        """
        dev = verts.device
        N   = self.N
        
        # ---------- 1. 计算所有当前边向量 e1 ----------
        e1 = verts[self.edge_cols] - verts[self.edge_rows]  # (E,3)
        
        # ---------- 2. 构建每个顶点的 S_v（全向量化） ----------
        # 对每条边：S_e = w_e * e0_e ⊗ e1_e  -> (E,3,3)
        E = self.edge_rows.shape[0]
        w = self.edge_weights.view(E, 1, 1)                 # (E,1,1)
        e0 = self.e0                                        # (E,3)
        e0_col = e0.unsqueeze(2)                            # (E,3,1)
        e1_row = e1.unsqueeze(1)                            # (E,1,3)
        S_e = w * (e0_col * e1_row)                         # (E,3,3)
        
        # 累加到每个顶点的 S_v，顶点索引用 edge_rows
        S_v = torch.zeros((N, 3, 3), device=dev, dtype=verts.dtype)
        S_v_flat = S_v.view(N, 9)
        S_e_flat = S_e.view(E, 9)
        S_v_flat.index_add_(0, self.edge_rows, S_e_flat)
        S_v = S_v_flat.view(N, 3, 3)                        # (N,3,3)
        
        # ---------- 3. batched SVD 估计所有 R_v ----------
        # 处理 valence==0 的点：直接置 S_v=I
        mask_iso = (self.valence == 0)
        if mask_iso.any():
            S_v[mask_iso] = self.I3
        
        U, Sigma, Vh = torch.linalg.svd(S_v, full_matrices=False)  # (N,3,3)
        R = Vh @ U.transpose(-2, -1)                              # (N,3,3)
        
        # 修正 det<0 的情况，确保是旋转阵（非 inplace 版本）
        det = torch.det(R)
        mask_neg = det < 0
        if mask_neg.any():
            # 创建新的 Vh 副本避免 inplace 操作
            Vh_corrected = Vh.clone()
            Vh_corrected[mask_neg, :, -1] *= -1.0
            R = Vh_corrected @ U.transpose(-2, -1)
        
        # 再次保证孤立点是单位阵
        if mask_iso.any():
            R = R.clone()  # 避免 inplace
            R[mask_iso] = self.I3
        
        # ---------- 4. E_arap（边级向量化） ----------
        # 对每条边，用其起点顶点的 R_i
        R_i = R[self.edge_rows]                                # (E,3,3)
        Re0 = torch.matmul(R_i, e0.unsqueeze(-1)).squeeze(-1)  # (E,3)
        diff_e = e1 - Re0                                      # (E,3)
        sq_e   = (diff_e ** 2).sum(-1)                         # (E,)
        per_edge = self.edge_weights * sq_e * self.area_v[self.edge_rows]
        E_arap = per_edge.sum() / float(N)
        
        # ---------- 5. E_smooth（顶点级向量化） ----------
        ell1   = self._laplacian_vec(verts)                   # (N,3)
        R_ell0 = torch.matmul(R, self.ell0.unsqueeze(-1)).squeeze(-1)  # (N,3)
        diff_lap = ell1 - R_ell0
        E_smooth = (self.area_v.view(-1, 1) * (diff_lap ** 2)).sum() / float(N)
        
        loss = self.lambda_arap * E_arap + self.lambda_smooth * E_smooth
        
        return loss
