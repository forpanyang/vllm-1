from typing import List, Tuple

import pytest
import torch


def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=dtype,
                                device='cuda')
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=dtype,
                                  device='cuda')
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches


def _quantize_kv_cache(fdata, is_cache=True):
    qtype = torch.int8
    if is_cache:
        head_dim = -2
        # fdata: [num_blocks, num_heads, head_dim, block_size]
    else:
        # fdata: [seqs, num_heads, head_dim]
        head_dim = -1

    fmax = torch.amax(fdata, dim=head_dim, keepdim=True)
    fmin = torch.amin(fdata, dim=head_dim, keepdim=True)
    # Compute params
    qmax = torch.tensor(torch.iinfo(qtype).max,
                        dtype=fdata.dtype, device=fdata.device)
    qmin = torch.tensor(torch.iinfo(qtype).min,
                        dtype=fdata.dtype, device=fdata.device)
    scale = (fmax - fmin) / (qmax - qmin)
    zero = fmin - qmin * scale
    # Quantize
    res_data = (fdata - zero) / scale
    qdata = torch.clamp(res_data, qmin, qmax).to(qtype)
    if is_cache:
        return qdata.contiguous(), \
            scale.transpose(-1, -2), zero.transpose(-1, -2)
    else:
        return qdata.contiguous(), scale, zero


def do_quantize_kv(key, value):
    q_key, k_scale, k_zero = _quantize_kv_cache(key, is_cache=False)
    q_value, v_scale, v_zero = _quantize_kv_cache(value, is_cache=False)
    quant_params = torch.cat([k_scale, k_zero, v_scale, v_zero], dim=-1)
    return q_key, q_value, quant_params


def do_quantize_kv_cache(key_cache, value_cache, is_cache=True):
    if not is_cache:
        return do_quantize_kv(key_cache, value_cache)

    num_blocks, num_kv_heads, head_size_div_x, block_size, x = key_cache.shape
    head_size = head_size_div_x * x
    key_cache_shape = (num_blocks, num_kv_heads, head_size, block_size)
    key_cache = key_cache.transpose(-1, -2).contiguous().view(key_cache_shape)
    q_key_cache_shape = (num_blocks, num_kv_heads, head_size//x, x, block_size)
    q_key_cache, k_scale, k_zero = _quantize_kv_cache(key_cache)
    q_key_cache = q_key_cache.view(
            q_key_cache_shape).transpose(-1, -2).contiguous()
    q_value_cache, v_scale, v_zero = _quantize_kv_cache(value_cache)
    quant_params = torch.cat([k_scale, k_zero, v_scale, v_zero], dim=-1)
    return q_key_cache, q_value_cache, quant_params


@pytest.fixture()
def quantize_kv_cache():
    return do_quantize_kv_cache


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches
