#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test client for the Rust KV Cache gRPC server."""

import sys

import grpc

# We need to generate Python code from the proto file first
# For now, we'll use a simple HTTP health check


def test_health_check():
    """Test if the server is responding to health checks."""
    # gRPC doesn't have built-in HTTP health checks, but we can test connection
    try:
        channel = grpc.insecure_channel('[::1]:50051')
        # Try to connect
        grpc.channel_ready_future(channel).result(timeout=5)
        print("✅ Successfully connected to gRPC server at [::1]:50051")
        return True
    except grpc.FutureTimeoutError:
        print("❌ Failed to connect to gRPC server - timeout")
        return False
    except Exception as e:
        print(f"❌ Failed to connect to gRPC server: {e}")
        return False


def main():
    """Run basic connectivity tests."""
    print("=" * 60)
    print("Testing Rust KV Cache gRPC Server")
    print("=" * 60)

    # Test connection
    if test_health_check():
        print("\n✅ Server is running and accepting connections")
    else:
        print("\n❌ Server is not responding")
        sys.exit(1)

    print("\nNote: To perform full testing, you need to:")
    print("1. Generate Python gRPC stubs from proto/kv_cache.proto")
    print("2. Import and use the generated client code")
    print("\nExample command to generate Python stubs:")
    print(
        "python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/kv_cache.proto"
    )


if __name__ == "__main__":
    main()
