# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from comfy.ldm.modules.diffusionmodules.model import vae_attention
from comfy.ldm.wan import vae_kernels

import comfy.ops
ops = comfy.ops.disable_weight_init

CACHE_T = 2


class CausalConv3d(ops.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
        if cache_list is not None:
            cache_x = cache_list[cache_idx]
            cache_list[cache_idx] = None

        if cache_x is None and x.shape[2] == 1:
            #Fast path - the op will pad for use by truncating the weight
            #and save math on a pile of zeros.
            return super().forward(_channels_last(x), autopad="causal_zero")

        if self._padding > 0:
            # Assemble [zeros, cache, x] in one channels_last_3d buffer: cuDNN then runs its NHWC kernels
            # directly. With NCDHW input it needs a layout-conversion workspace as large as the input, and
            # when that allocation fails near the memory peak it falls back to implicit_convolveNd_sgemm
            # (no tensor cores) and keeps that plan for the shape, several times slower.
            b, c, t, h, w = x.shape
            ct = 0 if cache_x is None else cache_x.shape[2]
            zeros = max(0, self._padding - ct)
            buf = torch.empty((b, c, zeros + ct + t, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
            if zeros:
                buf[:, :, :zeros].zero_()
            if ct:
                buf[:, :, zeros:zeros + ct].copy_(cache_x)
            buf[:, :, zeros + ct:].copy_(x)
            del cache_x
            x = buf
        else:
            x = _channels_last(x)

        return super().forward(x)


def _forward_normed(self, x, norm, cache_list, idx, residual=None):
    # silu(norm(x)) written by one kernel into the [zeros, cache, x] conv buffer; the cache keeps the
    # buffer's last CACHE_T frames (the normalized input), as the unfused path caches silu(norm(x)).
    cache = cache_list[idx]
    cache_list[idx] = None
    b, c, t, h, w = x.shape
    if cache is None and t == 1:
        xn = vae_kernels.rms_silu(x, norm.gamma, norm.scale, torch.empty_like(x, memory_format=torch.channels_last_3d))
        cache_list[idx] = xn
        out = ops.Conv3d.forward(self, xn, autopad="causal_zero")
    else:
        ct = 0 if cache is None else cache.shape[2]
        zeros = max(0, self._padding - ct)
        buf = torch.empty((b, c, zeros + ct + t, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
        if zeros:
            buf[:, :, :zeros].zero_()
        if ct:
            buf[:, :, zeros:zeros + ct].copy_(cache)
        del cache
        vae_kernels.rms_silu(x, norm.gamma, norm.scale, buf[:, :, zeros + ct:])
        cache_list[idx] = buf[:, :, -CACHE_T:].clone(memory_format=torch.channels_last_3d)
        out = ops.Conv3d.forward(self, buf)
        del buf
    if residual is not None:
        out += residual
    return out


CausalConv3d.forward_normed = _forward_normed


def _next_cache(x, old):
    # The last CACHE_T input frames of a causal conv, reaching into the previous cache when x is shorter.
    t = x.shape[2]
    if t >= CACHE_T or old is None or isinstance(old, str):
        return x[:, :, -CACHE_T:, :, :]
    return torch.cat([old[:, :, t - CACHE_T:], x], 2)


def _channels_last(x):
    return x if x.is_contiguous(memory_format=torch.channels_last_3d) else x.contiguous(memory_format=torch.channels_last_3d)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                ops.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                ops.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :]
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if self.mode.startswith('upsample') and x.shape[0] > 1:
            # One frame at a time: the nearest-upsampled input is 4x the frame, the decoder's memory peak.
            x = torch.cat([self.resample(x[i:i + 1]) for i in range(x.shape[0])])
        else:
            x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x
                else:

                    cache_x = x[:, :, -1:, :, :]
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x

                    deferred_x = feat_cache[idx + 1]
                    if deferred_x is not None:
                        x = torch.cat([deferred_x, x], 2)
                        feat_cache[idx + 1] = None

                    if x.shape[2] == 1 and not final:
                        feat_cache[idx + 1] = x
                        x = None

                feat_idx[0] += 2
        return None if x is None else _channels_last(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        if feat_cache is not None and vae_kernels.available(x):
            norm1, _, conv1, norm2, _, _, conv2 = self.residual
            h = conv1.forward_normed(x, norm1, feat_cache, feat_idx[0])
            feat_idx[0] += 1
            h = conv2.forward_normed(h, norm2, feat_cache, feat_idx[0], residual=self.shortcut(x))
            feat_idx[0] += 1
            return h
        old_x = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = _next_cache(x, feat_cache[idx])
                x = layer(x, cache_list=feat_cache, cache_idx=idx)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + self.shortcut(old_x)


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = ops.Conv2d(dim, dim * 3, 1)
        self.proj = ops.Conv2d(dim, dim, 1)
        self.optimized_attention = vae_attention()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = self.optimized_attention(q, k, v)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 input_channels=3,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, final=final)
                if x is None:
                    return None
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, final=final)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 output_channels=3,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.last_upsample3d = max((i for i, m in enumerate(upsamples) if isinstance(m, Resample) and m.mode == 'upsample3d'), default=-1)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, output_channels, 3, padding=1))

    def run_up(self, layer_idx, x_ref, feat_cache, feat_idx, out_chunks):
        x = x_ref[0]
        x_ref[0] = None
        if layer_idx >= len(self.upsamples):
            for layer in self.head:
                if isinstance(layer, CausalConv3d) and feat_cache is not None:
                    cache_x = _next_cache(x, feat_cache[feat_idx[0]])
                    x = layer(x, feat_cache[feat_idx[0]])
                    feat_cache[feat_idx[0]] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
            out_chunks.append(x)
            return

        layer = self.upsamples[layer_idx]
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

        # Full-resolution layers take one frame per call (their caches carry the two-frame history), which
        # halves the decoder's working set at the top resolution; earlier stages keep two-frame chunks.
        step = int(__import__("os").environ.get("WAN_TOP_STEP", "1")) if layer_idx == self.last_upsample3d else 2
        if isinstance(layer, Resample) and layer.mode == 'upsample3d' and x.shape[2] > step:
            for frame_idx in range(0, x.shape[2], step):
                self.run_up(
                    layer_idx + 1,
                    [x[:, :, frame_idx:frame_idx + step, :, :]],
                    feat_cache,
                    feat_idx.copy(),
                    out_chunks,
                )
            del x
            return

        next_x_ref = [x]
        del x
        self.run_up(layer_idx + 1, next_x_ref, feat_cache, feat_idx, out_chunks)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        out_chunks = []

        self.run_up(0, [x], feat_cache, feat_idx, out_chunks)
        return out_chunks


def count_cache_layers(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d) or (isinstance(m, Resample) and m.mode == 'downsample3d'):
            count += 1
    return count


class WanVAE(nn.Module):
    comfy_has_chunked_io = True

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 image_channels=3,
                 conv_out_channels=3,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, image_channels, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, conv_out_channels, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def encode(self, x, device=None):
        # chunked io: x may live on the CPU; each time slice moves to the device as it encodes
        if device is None:
            device = x.device
        conv_idx = [0]
        ## cache
        t = x.shape[2]
        t = 1 + ((t - 1) // 4) * 4
        iter_ = 1 + (t - 1) // 2
        feat_map = None
        if iter_ > 1:
            feat_map = [None] * count_cache_layers(self.encoder)
        ## 对encode输入的x，按时间拆分为1、2、2、2....(总帧数先按4N+1向下取整)
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :].to(device),
                    feat_cache=feat_map,
                    feat_idx=conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 2 * (i - 1):1 + 2 * i, :, :].to(device),
                    feat_cache=feat_map,
                    feat_idx=conv_idx,
                    final=(i == (iter_ - 1)))
                if out_ is None:
                    continue
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode_output_shape(self, shape):
        b, _, t, h, w = shape
        up = 2 ** len(self.temperal_upsample)
        return (b, self.decoder.head[-1].out_channels, 1 + 4 * (t - 1), h * up, w * up)

    def decode(self, z, output_buffer=None):
        # z: [b,c,t,h,w]
        if output_buffer is not None:
            # chunked io: finalized frames stream into the (CPU) buffer instead of a full clip on the GPU
            for i in range(z.shape[0]):
                pos = 0
                for out in self.decode_chunks(z[i:i + 1]):
                    output_buffer[i:i + 1, :, pos:pos + out.shape[2]].copy_(out)
                    pos += out.shape[2]
            return output_buffer
        full, pos = None, 0
        for out in self.decode_chunks(z):
            if full is None:
                full = torch.empty(self.decode_output_shape(z.shape), device=out.device, dtype=out.dtype)
            full[:, :, pos:pos + out.shape[2]].copy_(out)
            pos += out.shape[2]
            del out
        return full[:, :, :pos]

    def decode_chunks(self, z):
        iter_ = 1 + z.shape[2] // 2
        feat_map = None
        if iter_ > 1:
            feat_map = [None] * count_cache_layers(self.decoder)
        x = self.conv2(z)
        # yields decoded frame groups; callers write them into one output (a list plus torch.cat holds the clip twice)
        for i in range(iter_):
            conv_idx = [0]
            chunk = x[:, :, i:i + 1, :, :] if i == 0 else x[:, :, 1 + 2 * (i - 1):1 + 2 * i, :, :]
            yield from self.decoder(chunk, feat_cache=feat_map, feat_idx=conv_idx)
