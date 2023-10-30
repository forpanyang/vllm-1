import random

import pytest
import torch

from vllm import cache_ops

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
NUM_LAYERS = [5]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [8, 16, 32]
NUM_BLOCKS = [1024]  # Arbitrary values for testing
NUM_MAPPINGS = [32, 256]  # Arbitrary values for testing
SEEDS = [0]


@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_copy_blocks(
    kv_cache_factory,
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping = {}
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping[src] = [dst1, dst2]

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size,
                                                num_layers, num_heads,
                                                head_size, dtype, seed)

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Call the copy blocks kernel.
    cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

    # Run the reference implementation.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for cloned_key_cache in cloned_key_caches:
                cloned_key_cache[dst] = cloned_key_cache[src]
            for cloned_value_cache in cloned_value_caches:
                cloned_value_cache[dst] = cloned_value_cache[src]

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(value_caches,
                                               cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_copy_blocks_quantized(
    quantize_kv_cache,
    kv_cache_factory,
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping = {}
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping[src] = [dst1, dst2]

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size,
                                                num_layers, num_heads,
                                                head_size, dtype, seed)

    q_key_caches = []
    q_value_caches = []
    q_param_caches = []
    for k_cache, v_cache in zip(key_caches, value_caches):
        q_k_cache, q_v_cache, q_params = quantize_kv_cache(k_cache, v_cache)
        q_key_caches.append(q_k_cache)
        q_value_caches.append(q_v_cache)
        q_param_caches.append(q_params)

    # Clone the KV caches.
    cloned_key_caches = [key_cache.clone() for key_cache in q_key_caches]
    cloned_value_caches =  \
        [value_cache.clone() for value_cache in q_value_caches]
    cloned_param_caches = \
        [param_cache.clone() for param_cache in q_param_caches]

    # Call the copy blocks kernel.
    cache_ops.copy_blocks_quantized(
            q_key_caches, q_value_caches, q_param_caches, block_mapping)

    # Run the reference implementation.
    for src, dsts in block_mapping.items():
        for dst in dsts:
            for cloned_key_cache in cloned_key_caches:
                cloned_key_cache[dst] = cloned_key_cache[src]
            for cloned_value_cache in cloned_value_caches:
                cloned_value_cache[dst] = cloned_value_cache[src]
            for cloned_param_cache in cloned_param_caches:
                cloned_param_cache[dst] = cloned_param_cache[src]

    # import pdb;  pdb.set_trace()
    # Compare the results.
    for key_cache, cloned_key_cache in zip(q_key_caches, cloned_key_caches):
        assert torch.allclose(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(q_value_caches,
                                               cloned_value_caches):
        assert torch.allclose(value_cache, cloned_value_cache)
    for param_cache, cloned_param_cache in zip(q_param_caches,
                                               cloned_param_caches):
        assert torch.allclose(param_cache, cloned_param_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device="cuda")

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()

    # Call the reshape_and_cache kernel.
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                slot_mapping)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_reshape_and_cache_quantized(
    quantize_kv_cache,
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device="cuda")

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    _, key, value = qkv.unbind(dim=1)
    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    q_key_caches = []
    q_value_caches = []
    q_param_caches = []
    for k_cache, v_cache in zip(key_caches, value_caches):
        q_k_cache, q_v_cache, q_params = quantize_kv_cache(k_cache, v_cache)
        q_key_caches.append(q_k_cache)
        q_value_caches.append(q_v_cache)
        q_param_caches.append(q_params)

    key_cache, value_cache, param_cache = \
        q_key_caches[0], q_value_caches[0], q_param_caches[0]
    key, value, q_param = quantize_kv_cache(key, value, is_cache=False)

    # Clone the KV caches.
    cloned_key_cache = key_cache.clone()
    cloned_value_cache = value_cache.clone()
    cloned_param_cache = param_cache.clone()

    # Call the reshape_and_cache kernel.
    cache_ops.reshape_and_cache_quantized(
            key, value, q_param, key_cache, value_cache, param_cache,
            slot_mapping)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies[i]
        block_offset = block_offsets[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]
        cloned_param_cache[block_idx, :, block_offset] = q_param[i]

    assert torch.allclose(key_cache, cloned_key_cache)
    assert torch.allclose(value_cache, cloned_value_cache)
    assert torch.allclose(param_cache, cloned_param_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gather_cached_kv_quantized(
    quantize_kv_cache,
    kv_cache_factory,
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device="cuda")

    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    _, key, value = qkv.unbind(dim=1)
    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                num_heads, head_size, dtype,
                                                seed)
    q_key_caches = []
    q_value_caches = []
    q_param_caches = []
    for k_cache, v_cache in zip(key_caches, value_caches):
        q_k_cache, q_v_cache, q_params = quantize_kv_cache(k_cache, v_cache)
        q_key_caches.append(q_k_cache)
        q_value_caches.append(q_v_cache)
        q_param_caches.append(q_params)

    key_cache, value_cache, param_cache = \
        q_key_caches[0], q_value_caches[0], q_param_caches[0]
    key, value, q_param = quantize_kv_cache(key, value, is_cache=False)

    # Clone the KV caches.
    cloned_key = key.clone()
    cloned_value = value.clone()
    cloned_param = q_param.clone()

    # Call the reshape_and_cache kernel.
    cache_ops.reshape_and_cache_quantized(
            key, value, q_param, key_cache, value_cache, param_cache,
            slot_mapping)
    key.zero_()
    value.zero_()
    q_param.zero_()
    # Call the reshape_and_cache kernel.
    cache_ops.gather_cached_kv_quantized(
            key, value, q_param, key_cache, value_cache, param_cache,
            slot_mapping)

    assert torch.allclose(key, cloned_key)
    assert torch.allclose(value, cloned_value)
    assert torch.allclose(q_param, cloned_param)
