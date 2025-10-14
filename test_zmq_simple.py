#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple test of ZMQ communication with Rust server."""

import os
import subprocess
import time

import zmq

# Start the server
rust_binary = "vllm/v1/rust_core/target/release/vllm-core-server"
print(f"Starting server: {rust_binary}")

proc = subprocess.Popen(
    [rust_binary, "1000"],
    env={
        **os.environ, "ZMQ_PORT": "5557",
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
socket.connect("tcp://localhost:5557")
socket.setsockopt(zmq.RCVTIMEO, 2000)

try:
    # Test get_usage
    print("\nTesting get_usage...")
    socket.send_json({"method": "get_usage", "params": {}})
    response = socket.recv_json()
    print(f"Response: {response}")
    assert response["success"], f"Failed: {response.get('error')}"

    # Test reset_prefix_cache
    print("\nTesting reset_prefix_cache...")
    socket.send_json({"method": "reset_prefix_cache", "params": {}})
    response = socket.recv_json()
    print(f"Response: {response}")
    assert response["success"], f"Failed: {response.get('error')}"

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
