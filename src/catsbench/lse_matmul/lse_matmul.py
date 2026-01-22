import math
import torch
from torch.autograd import Function 
import triton
import triton.language as tl


MAX_BATCH_DIMS = 6

def _lse_autotune_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 16}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=2, num_stages=4),
    ]


@triton.autotune(
    configs=_lse_autotune_configs(),
    key=["M", "N", "K", "USE_EXP2"],
)
@triton.jit
def _lse_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr, b3: tl.constexpr, b4: tl.constexpr, b5: tl.constexpr,
    sa0, sa1, sa2, sa3, sa4, sa5,
    sb0, sb1, sb2, sb3, sb4, sb5,
    sc0, sc1, sc2, sc3, sc4, sc5,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BATCH_DIMS: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    off_a = tl.full((), 0, tl.int32)
    off_b = tl.full((), 0, tl.int32)
    off_c = tl.full((), 0, tl.int32)

    if BATCH_DIMS >= 1:
        i0 = pid_b % b0
        pid_b = pid_b // b0
        off_a += i0 * sa0
        off_b += i0 * sb0
        off_c += i0 * sc0
    if BATCH_DIMS >= 2:
        i1 = pid_b % b1
        pid_b = pid_b // b1
        off_a += i1 * sa1
        off_b += i1 * sb1
        off_c += i1 * sc1
    if BATCH_DIMS >= 3:
        i2 = pid_b % b2
        pid_b = pid_b // b2
        off_a += i2 * sa2
        off_b += i2 * sb2
        off_c += i2 * sc2
    if BATCH_DIMS >= 4:
        i3 = pid_b % b3
        pid_b = pid_b // b3
        off_a += i3 * sa3
        off_b += i3 * sb3
        off_c += i3 * sc3
    if BATCH_DIMS >= 5:
        i4 = pid_b % b4
        pid_b = pid_b // b4
        off_a += i4 * sa4
        off_b += i4 * sb4
        off_c += i4 * sc4
    if BATCH_DIMS >= 6:
        i5 = pid_b % b5
        pid_b = pid_b // b5
        off_a += i5 * sa5
        off_b += i5 * sb5
        off_c += i5 * sc5

    # tile indices
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
    if (pid_m * BLOCK_M >= M) or (pid_n * BLOCK_N >= N):
        return

    log2e = 1.4426950408889634
    ln2 = 0.6931471805599453

    mask_m = rm < M
    mask_n = rn < N
    m_acc = tl.full((BLOCK_M, BLOCK_N), -float("inf"), tl.float32)
    s_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):

        blk_max = tl.full((BLOCK_M, BLOCK_N), -float("inf"), tl.float32)
        blk_sum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for kk in tl.static_range(0, BLOCK_K):
            k = k0 + kk
            mk = k < K

            a_k = tl.load(
                A_ptr + off_a + rm[:, None] * stride_am + k * stride_ak,
                mask=mask_m[:, None] & mk,
                other=-float("inf"),
            ).to(tl.float32)

            b_k = tl.load(
                B_ptr + off_b + k * stride_bk + rn[None, :] * stride_bn,
                mask=mask_n[None, :] & mk,
                other=-float("inf"),
            ).to(tl.float32)

            v = a_k + b_k
            if USE_EXP2:
                v2 = v * log2e
                new = tl.maximum(blk_max, v2)
                blk_sum = blk_sum * \
                    tl.where(blk_max == float("-inf"), 0.0, tl.exp2(blk_max - new)) + \
                    tl.where(v2 == float("-inf"), 0.0, tl.exp2(v2 - new))
            else:
                new = tl.maximum(blk_max, v)
                blk_sum = blk_sum * \
                    tl.where(blk_max == float("-inf"), 0.0, tl.exp(blk_max - new)) + \
                    tl.where(v == float("-inf"), 0.0, tl.exp(v - new))
            blk_max = new

        new_m = tl.maximum(m_acc, blk_max)
        if USE_EXP2:
            s_acc = s_acc * tl.where(m_acc == float("-inf"), 0.0, tl.exp2(m_acc - new_m)) + \
                    blk_sum * tl.where(blk_max == float("-inf"), 0.0, tl.exp2(blk_max - new_m)) 
        else:
            s_acc = s_acc * tl.where(m_acc == float("-inf"), 0.0, tl.exp(m_acc - new_m)) + \
                    blk_sum * tl.where(blk_max == float("-inf"), 0.0, tl.exp(blk_max - new_m)) 
        m_acc = new_m

    if USE_EXP2:
        out = (m_acc + tl.log2(s_acc)) * ln2
    else:
        out = m_acc + tl.log(s_acc)
    out = tl.where(s_acc > 0, out, float("-inf"))

    tl.store(
        C_ptr + off_c + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


class LSEMatmul(Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, use_exp2: bool = True):
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        ctx.use_exp2 = use_exp2

        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f"Inner dim mismatch: a[..., M, K]={a.shape}, b[..., K, N]={b.shape}")

        if a.dtype != torch.float32:
            a = a.float()
        if b.dtype != torch.float32:
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
            return (
                triton.cdiv(m, meta["BLOCK_M"]), 
                triton.cdiv(n, meta["BLOCK_N"]), 
                math.prod(c_batch_dims) if c_batch_dims else 1
            )

        _lse_matmul_kernel[grid](
            a_view, b_view, c,
            m, n, k,
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

        log2e = 1.4426950408889634
        if use_exp2:
            t = grad_out * torch.exp2((-c) * log2e)
        else:
            t = grad_out * torch.exp(-c)

        if use_exp2:
            exp_a = torch.exp2(a_view * log2e)
            exp_b = torch.exp2(b_view * log2e)
        else:
            exp_a = torch.exp(a_view)
            exp_b = torch.exp(b_view)


        d_a_view = exp_a * torch.matmul(t, exp_b.transpose(-1, -2))
        d_b_view = exp_b * torch.matmul(exp_a.transpose(-1, -2), t)

        dA = d_a_view.sum_to_size(ctx.a_shape)
        dB = d_b_view.sum_to_size(ctx.b_shape)

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
    