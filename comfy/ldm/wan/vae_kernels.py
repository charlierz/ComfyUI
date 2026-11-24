# Fused RMS_norm (channel-first, F.normalize form) + SiLU for the Wan VAE on channels_last_3d tensors.
# One pass instead of norm, divide, scale, gamma and SiLU kernels; writes straight into a causal-conv buffer.

import torch

from comfy.cli_args import args
from comfy.model_management import cast_to

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


if triton is not None:

    @triton.jit
    def _rms_silu_kernel(
        x_ptr,
        g_ptr,
        y_ptr,
        n_rows,
        scale,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * ROWS + tl.arange(0, ROWS)[:, None].to(tl.int64)
        cols = tl.arange(0, BLOCK_C)[None, :]
        mask = (rows < n_rows) & (cols < C)
        x = tl.load(x_ptr + rows * C + cols, mask=mask, other=0.0).to(tl.float32)
        norm = tl.maximum(tl.sqrt(tl.sum(x * x, axis=1)), 1e-12)[:, None]
        g = tl.load(g_ptr + cols, mask=cols < C, other=0.0).to(tl.float32)
        y = x / norm * scale * g
        y = y * tl.sigmoid(y)
        tl.store(y_ptr + rows * C + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def _rows_view_ok(t):
    # batch 1, channels innermost, pixels packed: rows of C elements at stride C
    if t.dim() != 5 or t.shape[0] != 1 or not t.is_cuda:
        return False
    _, c, d, h, w = t.shape
    s = t.stride()
    return (
        (c == 1 or s[1] == 1)
        and (w == 1 or s[4] == c)
        and (h == 1 or s[3] == w * c)
        and (d == 1 or s[2] == h * w * c)
    )


def available(x):
    return (
        triton is not None
        and not args.disable_triton_backend
        and torch.version.cuda is not None
        and x.dtype in (torch.float16, torch.bfloat16)
        and _rows_view_ok(x)
        and torch.cuda.get_device_capability(x.device) >= (7, 5)
    )


def rms_silu(x, gamma, scale, out):
    """out = silu(normalize(x, dim=1) * scale * gamma); x and out are batch-1 channels_last_3d views."""
    assert _rows_view_ok(out) and out.shape == x.shape and out.dtype == x.dtype
    c = x.shape[1]
    n_rows = x.numel() // c
    block_c = triton.next_power_of_2(c)
    rows = max(1, 4096 // block_c)
    g = cast_to(gamma.reshape(-1), x.dtype, x.device)
    _rms_silu_kernel[(triton.cdiv(n_rows, rows),)](
        x, g, out, n_rows, float(scale), C=c, BLOCK_C=block_c, ROWS=rows
    )
