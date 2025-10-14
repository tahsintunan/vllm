#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for V1 Core with KV cache factory.

This test exercises the full path through the V1 scheduler with the KV cache manager.
You can switch between Python and Rust backends using the USE_RUST_KV environment variable.

Usage:
    # Test with Python backend (default)
    python test_kv_cache_factory_e2e.py

    # Test with Rust backend
    USE_RUST_KV=1 python test_kv_cache_factory_e2e.py
"""

import os
import time

# Force V1 Core
os.environ['VLLM_USE_V1'] = '1'

from vllm import LLM, SamplingParams

# Use a small model for faster testing
MODEL = "EleutherAI/pythia-14m"

# Test prompts
PROMPTS = [
    "Hello, my name is",
    "The quick brown fox",
    "Once upon a time",
    "In a galaxy far far away",
]


def test_basic_generation():
    """Test basic text generation with V1 Core."""
    print(f"\n{'='*60}")
    print(
        f"Testing basic generation with {'Rust' if os.getenv('USE_RUST_KV') == '1' else 'Python'} KV cache backend"
    )
    print(f"{'='*60}")

    llm = LLM(
        MODEL,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=256,
    )

    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,  # Deterministic for testing
    )

    outputs = llm.generate(PROMPTS, sampling_params)

    assert len(outputs) == len(PROMPTS)
    for i, output in enumerate(outputs):
        assert len(output.outputs) == 1
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids
        print(f"Prompt {i}: '{PROMPTS[i]}' -> '{generated_text.strip()}'")
        print(f"  Generated {len(generated_tokens)} tokens")
        assert len(generated_tokens) > 0
        assert len(generated_text) > 0


def test_prefix_caching():
    """Test prefix caching functionality with V1 Core."""
    print(f"\n{'='*60}")
    print(
        f"Testing prefix caching with {'Rust' if os.getenv('USE_RUST_KV') == '1' else 'Python'} KV cache backend"
    )
    print(f"{'='*60}")

    llm = LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        block_size=16,
    )

    # Use the same prompt twice to test caching
    prompt = "The capital of France is"
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

    # First generation - no cache hit
    start = time.time()
    outputs1 = llm.generate([prompt], sampling_params)
    time1 = time.time() - start

    # Second generation - should hit cache
    start = time.time()
    outputs2 = llm.generate([prompt], sampling_params)
    time2 = time.time() - start

    print(f"First generation: {time1:.3f}s")
    print(f"Second generation (cached): {time2:.3f}s")

    # Both should produce the same output (deterministic)
    assert outputs1[0].outputs[0].text == outputs2[0].outputs[0].text

    # Check if caching is reported (when available)
    if hasattr(outputs2[0], 'num_cached_tokens'):
        print(f"Cached tokens: {outputs2[0].num_cached_tokens}")
        assert outputs2[0].num_cached_tokens > 0


def test_chunked_prefill():
    """Test chunked prefill with V1 Core."""
    print(f"\n{'='*60}")
    print(
        f"Testing chunked prefill with {'Rust' if os.getenv('USE_RUST_KV') == '1' else 'Python'} KV cache backend"
    )
    print(f"{'='*60}")

    llm = LLM(
        MODEL,
        enforce_eager=True,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=32,
        max_num_batched_tokens=64,
        max_num_seqs=2,
    )

    # Use a longer prompt to trigger chunked prefill
    long_prompt = "Once upon a time, in a land far, far away, there lived a wise old wizard who knew many secrets about the ancient arts of magic and sorcery. He spent his days studying ancient tomes and scrolls, searching for the ultimate spell that would bring peace to the realm."

    sampling_params = SamplingParams(max_tokens=20, temperature=0.8)

    outputs = llm.generate([long_prompt], sampling_params)

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    print(f"Generated text: '{outputs[0].outputs[0].text.strip()}'")


def test_multiple_requests():
    """Test handling multiple concurrent requests."""
    print(f"\n{'='*60}")
    print(
        f"Testing multiple requests with {'Rust' if os.getenv('USE_RUST_KV') == '1' else 'Python'} KV cache backend"
    )
    print(f"{'='*60}")

    llm = LLM(
        MODEL,
        enforce_eager=True,
        max_num_seqs=8,
        max_num_batched_tokens=512,
    )

    # Generate with different sampling parameters
    prompts_and_params = [
        ("The weather today is", SamplingParams(max_tokens=15,
                                                temperature=0.5)),
        ("Python is a", SamplingParams(max_tokens=20, temperature=0.7)),
        ("Machine learning is", SamplingParams(max_tokens=25,
                                               temperature=0.9)),
        ("The future of AI", SamplingParams(max_tokens=30, temperature=0.3)),
    ]

    for prompt, params in prompts_and_params:
        outputs = llm.generate([prompt], params)
        assert len(outputs) == 1
        generated = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        print(f"Prompt: '{prompt}' -> {num_tokens} tokens")
        print(f"  Output: '{generated.strip()[:50]}...'")
        assert num_tokens > 0


def test_kv_cache_allocation_and_freeing():
    """Test KV cache allocation and freeing through multiple generations."""
    print(f"\n{'='*60}")
    print(
        f"Testing KV cache allocation/freeing with {'Rust' if os.getenv('USE_RUST_KV') == '1' else 'Python'} KV cache backend"
    )
    print(f"{'='*60}")

    llm = LLM(
        MODEL,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.3,  # Limit memory to test allocation/freeing
    )

    # Run multiple batches to exercise allocation and freeing
    for batch_num in range(3):
        print(f"\nBatch {batch_num + 1}:")
        prompts = [f"Batch {batch_num}, prompt {i}: Hello" for i in range(4)]
        sampling_params = SamplingParams(max_tokens=10, temperature=0.5)

        outputs = llm.generate(prompts, sampling_params)

        assert len(outputs) == 4
        for i, output in enumerate(outputs):
            assert len(output.outputs[0].token_ids) > 0
            print(
                f"  Request {i}: {len(output.outputs[0].token_ids)} tokens generated"
            )


def main():
    """Run all tests."""
    backend = "Rust" if os.getenv('USE_RUST_KV') == '1' else "Python"
    print(f"\n{'='*70}")
    print(f"Running V1 Core tests with {backend} KV cache backend")
    print(f"Model: {MODEL}")
    print(f"{'='*70}")

    try:
        test_basic_generation()
        test_prefix_caching()
        test_chunked_prefill()
        test_multiple_requests()
        test_kv_cache_allocation_and_freeing()

        print(f"\n{'='*70}")
        print(f"✅ All tests passed with {backend} backend!")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ Test failed with {backend} backend!")
        print(f"Error: {e}")
        print(f"{'='*70}")
        raise


if __name__ == "__main__":
    main()
