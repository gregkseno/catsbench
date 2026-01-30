import math
import torch
from torch.autograd import Function
import triton

from .forward_kernel import _lse_fwd_kernel
from .backward_kernel import _lse_bwd_dA_kernel, _lse_bwd_dB_kernel


MAX_BATCH_DIMS = 6

class LSEMatmul(Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, use_exp2: bool = True):
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f"Inner dim mismatch: a[..., M, K]={a.shape}, b[..., K, N]={b.shape}")

        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        ctx.use_exp2 = bool(use_exp2)

        a = a.float()
        b = b.float()

        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        c_batch_dims = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        if len(c_batch_dims) > MAX_BATCH_DIMS:
            raise ValueError(f"Too many batch dims ({len(c_batch_dims)}). Increase MAX_BATCH_DIMS.")
        sizes = list(c_batch_dims) + [1] * (MAX_BATCH_DIMS - len(c_batch_dims))

        a_view = a.expand(*c_batch_dims, m, k)
        b_view = b.expand(*c_batch_dims, k, n)
        c = torch.empty((*c_batch_dims, m, n), device=a.device, dtype=torch.float32)

        sa = a_view.stride()[:-2]
        sb = b_view.stride()[:-2]
        sc = c.stride()[:-2]

        sa_p = list(sa) + [0] * (MAX_BATCH_DIMS - len(sa))
        sb_p = list(sb) + [0] * (MAX_BATCH_DIMS - len(sb))
        sc_p = list(sc) + [0] * (MAX_BATCH_DIMS - len(sc))

        def grid(meta):
            gm = triton.cdiv(m, meta["BLOCK_M"])
            gn = triton.cdiv(n, meta["BLOCK_N"])
            gb = math.prod(c_batch_dims) if c_batch_dims else 1
            return (gb, gm, gn)

        _lse_fwd_kernel[grid](
            a_view, b_view, c,
            M=m, N=n, K=k,
            b0=sizes[0], b1=sizes[1], b2=sizes[2], b3=sizes[3], b4=sizes[4], b5=sizes[5],
            sa0=sa_p[0], sa1=sa_p[1], sa2=sa_p[2], sa3=sa_p[3], sa4=sa_p[4], sa5=sa_p[5],
            sb0=sb_p[0], sb1=sb_p[1], sb2=sb_p[2], sb3=sb_p[3], sb4=sb_p[4], sb5=sb_p[5],
            sc0=sc_p[0], sc1=sc_p[1], sc2=sc_p[2], sc3=sc_p[3], sc4=sc_p[4], sc5=sc_p[5],
            stride_am=a_view.stride(-2), stride_ak=a_view.stride(-1),
            stride_bk=b_view.stride(-2), stride_bn=b_view.stride(-1),
            stride_cm=c.stride(-2),      stride_cn=c.stride(-1),
            BATCH_DIMS=len(c_batch_dims),
            USE_EXP2=use_exp2,
        )
        ctx.save_for_backward(a_view, b_view, c)
        return c
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a_view, b_view, c = ctx.saved_tensors
        use_exp2 = ctx.use_exp2

        # Use float32 for stability
        g = grad_out.float()

        # Allocate full grads in expanded shapes, then sum_to_size like before
        dA_full = torch.empty_like(a_view, dtype=torch.float32)
        dB_full = torch.empty_like(b_view, dtype=torch.float32)

        m, k, n = a_view.shape[-2], a_view.shape[-1], b_view.shape[-1]
        c_batch_dims = c.shape[:-2]
        sizes = list(c_batch_dims) + [1] * (MAX_BATCH_DIMS - len(c_batch_dims))

        sa = a_view.stride()[:-2]
        sb = b_view.stride()[:-2]
        sc = c.stride()[:-2]
        sg = g.stride()[:-2]
        sda = dA_full.stride()[:-2]
        sdb = dB_full.stride()[:-2]

        sa_p = list(sa) + [0] * (MAX_BATCH_DIMS - len(sa))
        sb_p = list(sb) + [0] * (MAX_BATCH_DIMS - len(sb))
        sc_p = list(sc) + [0] * (MAX_BATCH_DIMS - len(sc))
        sg_p = list(sg) + [0] * (MAX_BATCH_DIMS - len(sg))
        sda_p = list(sda) + [0] * (MAX_BATCH_DIMS - len(sda))
        sdb_p = list(sdb) + [0] * (MAX_BATCH_DIMS - len(sdb))

        gb = math.prod(c_batch_dims) if c_batch_dims else 1

        def grid_dA(meta):
            gm = triton.cdiv(m, meta["BLOCK_M"])
            gk = triton.cdiv(k, meta["BLOCK_K"])
            return (gb, gm, gk)

        def grid_dB(meta):
            gk = triton.cdiv(k, meta["BLOCK_K"])
            gn = triton.cdiv(n, meta["BLOCK_N"])
            return (gb, gk, gn)

        _lse_bwd_dA_kernel[grid_dA](
            a_view, b_view, c, g,
            dA_full,
            M=m, N=n, K=k,
            b0=sizes[0], b1=sizes[1], b2=sizes[2], b3=sizes[3], b4=sizes[4], b5=sizes[5],
            sa0=sa_p[0], sa1=sa_p[1], sa2=sa_p[2], sa3=sa_p[3], sa4=sa_p[4], sa5=sa_p[5],
            sb0=sb_p[0], sb1=sb_p[1], sb2=sb_p[2], sb3=sb_p[3], sb4=sb_p[4], sb5=sb_p[5],
            sc0=sc_p[0], sc1=sc_p[1], sc2=sc_p[2], sc3=sc_p[3], sc4=sc_p[4], sc5=sc_p[5],
            sg0=sg_p[0], sg1=sg_p[1], sg2=sg_p[2], sg3=sg_p[3], sg4=sg_p[4], sg5=sg_p[5],
            sda0=sda_p[0], sda1=sda_p[1], sda2=sda_p[2], sda3=sda_p[3], sda4=sda_p[4], sda5=sda_p[5],
            stride_am=a_view.stride(-2), stride_ak=a_view.stride(-1),
            stride_bk=b_view.stride(-2), stride_bn=b_view.stride(-1),
            stride_cm=c.stride(-2),      stride_cn=c.stride(-1),
            stride_gm=g.stride(-2),      stride_gn=g.stride(-1),
            stride_dam=dA_full.stride(-2), stride_dak=dA_full.stride(-1),
            BATCH_DIMS=len(c_batch_dims),
            USE_EXP2=use_exp2,
        )

        _lse_bwd_dB_kernel[grid_dB](
            a_view, b_view, c, g,
            dB_full,
            M=m, N=n, K=k,
            b0=sizes[0], b1=sizes[1], b2=sizes[2], b3=sizes[3], b4=sizes[4], b5=sizes[5],
            sa0=sa_p[0], sa1=sa_p[1], sa2=sa_p[2], sa3=sa_p[3], sa4=sa_p[4], sa5=sa_p[5],
            sb0=sb_p[0], sb1=sb_p[1], sb2=sb_p[2], sb3=sb_p[3], sb4=sb_p[4], sb5=sb_p[5],
            sc0=sc_p[0], sc1=sc_p[1], sc2=sc_p[2], sc3=sc_p[3], sc4=sc_p[4], sc5=sc_p[5],
            sg0=sg_p[0], sg1=sg_p[1], sg2=sg_p[2], sg3=sg_p[3], sg4=sg_p[4], sg5=sg_p[5],
            sdb0=sdb_p[0], sdb1=sdb_p[1], sdb2=sdb_p[2], sdb3=sdb_p[3], sdb4=sdb_p[4], sdb5=sdb_p[5],
            stride_am=a_view.stride(-2), stride_ak=a_view.stride(-1),
            stride_bk=b_view.stride(-2), stride_bn=b_view.stride(-1),
            stride_cm=c.stride(-2),      stride_cn=c.stride(-1),
            stride_gm=g.stride(-2),      stride_gn=g.stride(-1),
            stride_dbk=dB_full.stride(-2), stride_dbn=dB_full.stride(-1),
            BATCH_DIMS=len(c_batch_dims),
            USE_EXP2=use_exp2,
        )

        dA = dA_full.sum_to_size(ctx.a_shape)
        dB = dB_full.sum_to_size(ctx.b_shape)
        return dA, dB, None

def lse_matmul(
    a: torch.Tensor, 
    b: torch.Tensor,
    use_exp2: bool = True
) -> torch.Tensor:
    
    if a.is_cuda and b.is_cuda:
        return LSEMatmul.apply(a, b, use_exp2)

    # CPU fallback
    return torch.logsumexp(
        a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2
    )
    