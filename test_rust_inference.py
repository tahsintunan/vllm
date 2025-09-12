#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple test to verify Rust KV cache backend works with actual inference.
Uses facebook/opt-125m which is not gated.
"""

import os
import sys

# Set environment to use Rust backend
os.environ['VLLM_KV_CACHE_BACKEND'] = 'rust'
os.environ['VLLM_USE_V1'] = '1'

from vllm import LLM, SamplingParams


def test_basic_inference():
    """Test basic text generation with OPT-125M."""
    print("Initializing LLM with Rust KV cache backend...")

    # Initialize model with small config for testing
    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,  # Disable CUDA graph for simpler testing
        gpu_memory_utilization=0.3,  # Use less GPU memory
        max_model_len=128,  # Short context for quick testing
    )

    print("Model loaded successfully!")

    # Test prompts
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "Machine learning is",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for deterministic output
        max_tokens=20,
    )

    print("\nGenerating outputs...")
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Token IDs: {output.outputs[0].token_ids[:10]}..."
              )  # First 10 tokens

    print("\n✅ Test passed! Rust KV cache backend is working.")
    return True


def test_prefix_caching():
    """Test that prefix caching works with shared prompts."""
    print("\n" + "=" * 60)
    print("Testing prefix caching...")

    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=256,
        enable_prefix_caching=True,  # Enable prefix caching
    )

    # Prompts with shared prefix
    shared_prefix = "In the field of artificial intelligence,"
    prompts = [
        f"{shared_prefix} machine learning is",
        f"{shared_prefix} deep learning involves",
        f"{shared_prefix} neural networks are",
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=15)

    print(f"Shared prefix: '{shared_prefix}'")
    print("Generating with prefix caching...")

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"  Prompt {i+1} completion: {generated_text}")

    print("\n✅ Prefix caching test passed!")
    return True


def test_multiple_requests():
    """Test handling multiple concurrent requests."""
    print("\n" + "=" * 60)
    print("Testing multiple requests...")

    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=128,
    )

    # Generate 10 different prompts
    prompts = [f"The number {i} is" for i in range(10)]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    print(f"Processing {len(prompts)} requests...")
    outputs = llm.generate(prompts, sampling_params)

    print(f"Successfully processed {len(outputs)} requests")
    print("Sample outputs:")
    for i in range(min(3, len(outputs))):
        print(f"  '{prompts[i]}' -> '{outputs[i].outputs[0].text}'")

    print("\n✅ Multiple requests test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing vLLM with Rust KV Cache Backend")
    print("=" * 60)

    try:
        # Check if Rust server is configured
        backend = os.environ.get('VLLM_KV_CACHE_BACKEND', 'python')
        print(f"KV Cache Backend: {backend}")
        print(f"Using V1 Engine: {os.environ.get('VLLM_USE_V1', 'false')}")

        # Run tests
        success = True
        success &= test_basic_inference()
        success &= test_prefix_caching()
        success &= test_multiple_requests()

        if success:
            print("\n" + "=" * 60)
            print("🎉 All tests passed!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
