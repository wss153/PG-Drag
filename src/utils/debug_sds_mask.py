"""
Debug utilities for checking SDS gradient masking effectiveness.
"""

import torch


@torch.no_grad()
def debug_check_sds_vertex_grad(
    curr_v: torch.Tensor,         # (N, 3) current vertices (from Poisson)
    curr_f: torch.Tensor,         # (F, 3) faces
    vertex_weight: torch.Tensor,  # (N,) vertex SDS weights: Core=0, Band/Far>0
    guidance,                     # SDS guidance callable
    guidance_kwargs: dict,        # kwargs for guidance (without weight_map)
    sds_weight_map: torch.Tensor, # (1, 1, H, W) rendered weight map
    core_thresh: float = 1e-4,
):
    """
    检查：在当前设置下，SDS 对顶点的梯度是否在 Core 区被屏蔽。
    
    这个函数会：
    1. 克隆当前顶点并设为 requires_grad
    2. 只计算 SDS loss 并反传
    3. 比较 Core 顶点 vs 非 Core 顶点的梯度大小
    
    如果屏蔽生效，应该看到：
      - |grad|_core ≈ 1e-8 (浮点噪声级别)
      - |grad|_noncore ≈ 1e-3~1e-2 (正常梯度)
    
    如果两者同一量级，说明 mask 没生效。
    """
    
    # 1) 创建独立的顶点变量（叶子节点，可求梯度）
    v_test = curr_v.detach().clone().requires_grad_(True)
    
    # 2) 调用 guidance 计算 SDS loss（传入 weight_map）
    try:
        # Merge weight_map into kwargs
        full_kwargs = {**guidance_kwargs, 'weight_map': sds_weight_map}
        loss_sds = guidance(**full_kwargs)
        
        # 只对 SDS loss 反传
        loss_sds.backward()
        
        # 3) 获取顶点梯度
        if v_test.grad is None:
            print("[DEBUG-SDS-MASK] ❌ v_test.grad is None - gradient computation failed")
            return
        
        gv = v_test.grad.norm(dim=1)  # (N,) L2 norm of gradient per vertex
        
        # 4) 分析 Core vs 非 Core
        core_mask = vertex_weight <= core_thresh
        non_mask = ~core_mask
        
        if core_mask.sum() == 0:
            print("[DEBUG-SDS-MASK] ⚠️  No core vertices found (check vertex_weight / core_thresh)")
            return
        
        # 统计
        n_core = core_mask.sum().item()
        n_non = non_mask.sum().item()
        
        core_mean = gv[core_mask].mean().item()
        core_max = gv[core_mask].max().item()
        non_mean = gv[non_mask].mean().item()
        non_max = gv[non_mask].max().item()
        
        # 打印结果
        print("\n" + "=" * 80)
        print("[DEBUG-SDS-MASK] Vertex Gradient Analysis")
        print("=" * 80)
        print(f"  Core vertices (masked):    {n_core:5d}")
        print(f"  Non-core vertices:         {n_non:5d}")
        print(f"\n  Gradient norms:")
        print(f"    Core region:")
        print(f"      - Mean: {core_mean:.3e}")
        print(f"      - Max:  {core_max:.3e}")
        print(f"    Non-core region:")
        print(f"      - Mean: {non_mean:.3e}")
        print(f"      - Max:  {non_max:.3e}")
        
        # 判断
        ratio = core_mean / (non_mean + 1e-12)
        print(f"\n  Ratio: core_mean / non_mean = {ratio:.3e}")
        
        if ratio < 1e-4:
            print("\n  ✅ PASSED: Core gradients are ~0, masking IS working!")
        elif ratio < 0.1:
            print("\n  ⚠️  PARTIAL: Core gradients reduced but not zero")
            print("     Possible issues: weight_map interpolation, mixed precision")
        else:
            print("\n  ❌ FAILED: Core gradients are NOT masked!")
            print("     Check: weight_map passed to guidance? Applied to per-pixel loss?")
        
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"[DEBUG-SDS-MASK] ❌ Error during gradient check: {e}")
        import traceback
        traceback.print_exc()

