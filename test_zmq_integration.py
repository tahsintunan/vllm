#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the ZMQ integration between Python and Rust KV cache manager."""

import os
import sys
import time

# Force V1 and Rust KV cache
os.environ['VLLM_USE_V1'] = '1'
os.environ['USE_RUST_KV'] = '1'
os.environ['RUST_LOG'] = 'info'
os.environ['ZMQ_PORT'] = '5556'  # Use different port to avoid conflicts

# Set the Rust binary path
rust_binary = os.path.join(
    os.path.dirname(__file__),
    "vllm/v1/rust_core/target/release/vllm-core-server")
os.environ['RUST_KV_CACHE_BINARY'] = rust_binary

print("=" * 70)
print("Testing ZMQ Integration with Rust KV Cache Manager")
print("=" * 70)
print(f"Rust binary: {rust_binary}")
print("ZMQ port: 5556")

# Check if binary exists
if not os.path.exists(rust_binary):
    print(f"ERROR: Rust binary not found at {rust_binary}")
    print(
        "Please build it with: cd vllm/v1/rust_core && cargo build --release")
    sys.exit(1)

print("\n1. Testing direct ZMQ communication...")
print("-" * 40)

import zmq

# Test direct ZMQ communication
context = zmq.Context()
socket = context.socket(zmq.REQ)

# Start the server manually for testing
import subprocess

print("Starting Rust server...")
proc = subprocess.Popen(
    [rust_binary, "1000"],
    env={
        **os.environ, "ZMQ_PORT": "5556"
    },
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for server to start
time.sleep(2)

# Connect to ZMQ
socket.connect("tcp://localhost:5556")
socket.setsockopt(zmq.RCVTIMEO, 5000)

try:
    # Test get_usage
    print("Testing get_usage...")
    request = {"method": "get_usage", "params": {}}
    socket.send_json(request)
    response = socket.recv_json()
    print(f"  Response: {response}")
    assert response.get("success",
                        False), f"get_usage failed: {response.get('error')}"
    print(f"  ✓ Usage: {response.get('usage', 0)}")

    # Test reset_prefix_cache
    print("\nTesting reset_prefix_cache...")
    request = {"method": "reset_prefix_cache", "params": {}}
    socket.send_json(request)
    response = socket.recv_json()
    print(f"  Response: {response}")
    assert response.get(
        "success",
        False), f"reset_prefix_cache failed: {response.get('error')}"
    print("  ✓ Prefix cache reset")

    print("\n✅ Direct ZMQ communication working!")

except zmq.error.Again:
    print("❌ ZMQ timeout - server may not be responding")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    # Shutdown
    try:
        request = {"method": "shutdown", "params": {}}
        socket.send_json(request)
        socket.recv_json()
    except:
        pass

    socket.close()
    context.term()
    proc.terminate()
    proc.wait(timeout=5)

print("\n2. Testing via vLLM factory...")
print("-" * 40)

# Now test via vLLM factory
try:
    from vllm.v1.core.kv_cache_backend_factory import create_kv_cache_manager

    # Create a mock KVCacheConfig
    class MockKVCacheConfig:

        def __init__(self):
            self.num_blocks = 100
            self.kv_cache_groups = []

    kv_config = MockKVCacheConfig()

    print("Creating Rust KV cache manager via factory...")
    manager = create_kv_cache_manager(
        kv_cache_config=kv_config,
        max_model_len=2048,
        enable_caching=True,
    )

    print(f"Manager type: {type(manager).__name__}")

    # Test some methods
    print("\nTesting manager.get_usage()...")
    usage = manager.get_usage()
    print(f"  Usage: {usage}")

    print("\nTesting manager.reset_prefix_cache()...")
    result = manager.reset_prefix_cache()
    print(f"  Result: {result}")

    print("\n✅ vLLM factory integration working!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
