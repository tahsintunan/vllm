# Rust KV Cache Server

A high-performance key-value cache server written in Rust for vLLM, using Protocol Buffers for efficient IPC communication with Python clients.

## Overview

This server manages block allocation and caching for vLLM's KV cache system. It communicates with Python clients via ZeroMQ sockets using Protocol Buffers serialization, providing better performance than the pure Python implementation.

## Architecture

- **Transport**: ZeroMQ REQ-REP pattern over Unix domain sockets
- **Serialization**: Protocol Buffers (protobuf)
- **Memory Management**: Cache-line aligned blocks with reference counting
- **Data Structures**: Optimized free block queue and hash map for cached blocks

## Prerequisites

- Rust toolchain (1.70+)
- Protocol Buffers compiler (`protoc`)
- ZeroMQ library

### Installing Dependencies

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt-get install protobuf-compiler
sudo apt-get install libzmq3-dev
```

## Building the Server

### 1. Compile Protocol Buffers

The protobuf schema is **automatically compiled** during the build process via `build.rs`. You don't need to manually compile it for Rust.

The build script (`build.rs`) uses `prost_build` to:
1. Call `protoc` internally
2. Generate Rust code from `kv_cache.proto`
3. Place the generated code in the build output directory
4. Include it in the compilation

For the Python client, if the generated file is missing or needs updating:
```bash
# Run from the rust_kv_cache_server directory
cd rust_kv_cache_server

# Generate Python protobuf code (only if vllm/v1/core/kv_cache_pb2.py is missing)
protoc --python_out=../vllm/v1/core/ kv_cache.proto
```

### 2. Build the Rust Server

```bash
cargo build --release
```

### 3. Run the Server

```bash
cargo run --release

# Or run the compiled binary
./target/release/kv_cache_server

# The server accepts an optional argument for number of blocks (default: from env or 100)
cargo run --release -- 1000
```

## Configuration

The server can be configured via environment variables:

- `KV_CACHE_NUM_GPU_BLOCKS`: Number of GPU blocks to allocate (default: 100)
- `KV_CACHE_ENABLE_CACHING`: Enable block caching (default: false)
- `KV_CACHE_SOCKET_PATH`: ZeroMQ socket path (default: `ipc:///tmp/kv_cache.sock`)

Example:
```bash
KV_CACHE_NUM_GPU_BLOCKS=5000 KV_CACHE_ENABLE_CACHING=true cargo run --release
```

## Protocol Buffer Schema

The `kv_cache.proto` file defines the following operations:

- **Ping**: Health check
- **Allocate**: Allocate blocks for a request
- **Free**: Free blocks by request ID
- **FreeBlocksById**: Free specific blocks by ID
- **Touch**: Increment reference count for blocks
- **GetCachedBlock**: Retrieve cached block by hash
- **CacheFullBlocks**: Cache blocks with their hashes
- **GetStats**: Get memory usage statistics
- **GetAllFreeBlocks**: List all free blocks
- **ResetPrefixCache**: Reset the prefix cache
- **Shutdown**: Gracefully shutdown the server

## Python Integration

The server is designed to work with vLLM's Python client located at `vllm/v1/core/rust_kv_cache_client.py`. The client automatically starts the server when initialized.

```python
from vllm.v1.core.kv_cache_backend_factory import create_kv_cache_backend

# Use Rust backend
import os
os.environ['VLLM_KV_CACHE_BACKEND'] = 'rust'

backend = create_kv_cache_backend(
    num_gpu_blocks=1000,
    enable_caching=True,
    enable_kv_cache_events=False
)
```

## Development

### Benchmarking

Use the benchmark script to compare Python vs Rust performance:

```bash
VLLM_USE_V1=1 python3 benchmarks/throughput_benchmark.py
```

## Troubleshooting

### Server fails to start
- Check if the socket path is accessible
- Ensure no other process is using the same socket
- Verify ZeroMQ is installed correctly

### Protocol buffer compilation errors
- Ensure `protoc` is installed and in PATH
- Check that `prost-build` version matches `prost` version in Cargo.toml

### Connection timeouts
- Increase the startup delay in the Python client
- Check firewall/security settings for Unix domain sockets
- Verify the server is running and listening on the correct socket

## License

Part of the vLLM project. See the main project LICENSE file for details.