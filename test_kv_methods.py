#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test KV cache methods via ZMQ."""

import os
import subprocess
import time

import zmq

# Start the server
rust_binary = "vllm/v1/rust_core/target/release/vllm-core-server"
print(f"Starting server: {rust_binary}")

proc = subprocess.Popen(
    [rust_binary, "100"],  # 100 blocks
    env={
        **os.environ, "ZMQ_PORT": "5558",
        "RUST_LOG": "info"
    },
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Wait for server to start
time.sleep(2)

# Connect via ZMQ
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5558")
socket.setsockopt(zmq.RCVTIMEO, 2000)

try:
    # Test get_usage
    print("\n1. Testing get_usage...")
    socket.send_json({"method": "get_usage", "params": {}})
    response = socket.recv_json()
    print(f"   Response: {response}")
    assert response["success"], f"Failed: {response.get('error')}"
    print(f"   ✓ Usage: {response.get('usage', 0)}")

    # Test get_num_free_blocks
    print("\n2. Testing get_num_free_blocks...")
    socket.send_json({"method": "get_num_free_blocks", "params": {}})
    response = socket.recv_json()
    print(f"   Response: {response}")
    assert response["success"], f"Failed: {response.get('error')}"
    print(f"   ✓ Free blocks: {response.get('num_free_blocks', 0)}")

    # Test reset_prefix_cache
    print("\n3. Testing reset_prefix_cache...")
    socket.send_json({"method": "reset_prefix_cache", "params": {}})
    response = socket.recv_json()
    print(f"   Response: {response}")
    assert response["success"], f"Failed: {response.get('error')}"
    print("   ✓ Prefix cache reset")

    # Test unimplemented method
    print("\n4. Testing unimplemented method...")
    socket.send_json({
        "method": "allocate_slots",
        "params": {
            "request_id": "test"
        }
    })
    response = socket.recv_json()
    print(f"   Response: {response}")
    if not response["success"]:
        print(f"   ✓ Expected error: {response.get('error')}")

    print("\n✅ All tests passed!")

finally:
    # Shutdown
    print("\nShutting down...")
    try:
        socket.send_json({"method": "shutdown", "params": {}})
        response = socket.recv_json()
        print(f"Shutdown response: {response}")
    except:
        pass

    socket.close()
    context.term()

    # Terminate process
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()
        proc.wait()

    print("Done!")
