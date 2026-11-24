import pytest
import torch
import torch.nn.functional as F

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.wan import vae, vae_kernels


def init_convs(model):
    rng = torch.Generator().manual_seed(42)
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
            torch.nn.init.normal_(layer.weight, std=0.03, generator=rng)
            torch.nn.init.normal_(layer.bias, std=0.03, generator=rng)
    return model.eval()


@pytest.fixture
def wan_vae():
    return init_convs(vae.WanVAE(dim=4, z_dim=2, num_res_blocks=1))


def stock_conv(layer, x, cache_x=None, cache_list=None, cache_idx=None):
    if cache_list is not None:
        cache_x = cache_list[cache_idx]
        cache_list[cache_idx] = None
    if layer._padding:
        if cache_x is not None:
            x = torch.cat((cache_x, x), dim=2)
        missing = layer._padding - (0 if cache_x is None else cache_x.shape[2])
        x = F.pad(x, (0, 0, 0, 0, missing, 0))
    return F.conv3d(x, layer.weight, layer.bias, layer.stride, layer.padding)


@pytest.mark.parametrize("frames, cached", [(1, 0), (1, 1), (1, 2), (2, 0), (2, 2)])
@torch.no_grad()
def test_causal_conv_matches_left_padding(frames, cached):
    conv = init_convs(vae.CausalConv3d(4, 8, 3, padding=1))
    x = torch.randn(1, 4, frames, 5, 7)
    history = torch.randn(1, 4, cached, 5, 7) if cached else None
    torch.testing.assert_close(conv(x, history), stock_conv(conv, x, history))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_fused_residual_matches_unfused(dtype, monkeypatch):
    block = init_convs(vae.ResidualBlock(8, 16)).to(device="cuda", dtype=dtype)
    x = torch.randn(1, 8, 5, 4, 5, device="cuda", dtype=dtype).contiguous(
        memory_format=torch.channels_last_3d
    )
    if not vae_kernels.available(x):
        pytest.skip("requires a supported CUDA device and Triton")

    def run():
        cache = [None, None]
        return torch.cat(
            [block(chunk, cache, [0]) for chunk in x.split((1, 2, 1, 1), dim=2)], dim=2
        )

    rms_silu = vae_kernels.rms_silu
    calls = []

    def record_fusion(*args):
        calls.append(args[0].shape[2])
        return rms_silu(*args)

    monkeypatch.setattr(vae_kernels, "rms_silu", record_fusion)
    fused = run()
    assert calls == [1, 1, 2, 2, 1, 1, 1, 1]
    monkeypatch.setattr(vae_kernels, "triton", None)
    unfused = run()
    atol, rtol = (0.002, 0.004) if dtype == torch.float16 else (0.018, 0.025)
    torch.testing.assert_close(fused, unfused, atol=atol, rtol=rtol)


@torch.no_grad()
def test_single_frame_matches_stock_convolutions(wan_vae, monkeypatch):
    x = torch.randn(2, 3, 1, 24, 32)
    z = wan_vae.encode(x)
    decoded = wan_vae.decode(z)
    assert z.shape == (2, 2, 1, 3, 4)
    assert decoded.shape == x.shape

    monkeypatch.setattr(vae.CausalConv3d, "forward", stock_conv)
    torch.testing.assert_close(z, wan_vae.encode(x))
    torch.testing.assert_close(decoded, wan_vae.decode(z))


@pytest.mark.parametrize("latent_frames", [3, 4])
@torch.no_grad()
def test_one_frame_decode_matches_two_frame_schedule(
    wan_vae, latent_frames, monkeypatch
):
    z = torch.randn(1, 2, latent_frames, 3, 4)
    out = wan_vae.decode(z)
    assert out.shape == (1, 3, 1 + 4 * (latent_frames - 1), 24, 32)

    monkeypatch.setattr(wan_vae.decoder, "last_upsample3d", -1)
    torch.testing.assert_close(out, wan_vae.decode(z), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
@pytest.mark.parametrize("latent_frames", [1, 4])
@torch.no_grad()
def test_decode_into_cpu_buffer_matches_full_decode(wan_vae, device, latent_frames):
    dtype = torch.float16 if device == "cuda" else torch.float32
    wan_vae.to(device=device, dtype=dtype)
    z = torch.randn(2, 2, latent_frames, 3, 4, device=device, dtype=dtype)
    out = torch.full(wan_vae.decode_output_shape(z.shape), float("nan"), device="cpu")
    assert wan_vae.decode(z, output_buffer=out) is out
    expected = torch.cat([wan_vae.decode(sample) for sample in z.split(1)])
    torch.testing.assert_close(out, expected.cpu().float())


@pytest.mark.parametrize("frames", [1, 9])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_encode_cpu_input_matches_device_input(wan_vae, frames):
    wan_vae.to(device="cuda", dtype=torch.float16)
    x = torch.randn(1, frames, 24, 32, 3, dtype=torch.float16).permute(0, 4, 1, 2, 3)
    expected = wan_vae.encode(x.cuda())
    out = wan_vae.encode(x, device="cuda")
    torch.testing.assert_close(out, expected)
