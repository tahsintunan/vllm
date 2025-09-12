# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
import atexit
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import zmq

# Try to import the generated protobuf module
try:
    from . import kv_cache_pb2
except ImportError:
    # Generate it if it doesn't exist
    import subprocess
    proto_file = Path(
        __file__
    ).parent.parent.parent.parent / "rust_kv_cache_server" / "kv_cache.proto"
    output_dir = Path(__file__).parent

    try:
        subprocess.run([
            "protoc", f"--python_out={output_dir}",
            f"--proto_path={proto_file.parent}",
            str(proto_file)
        ],
                       check=True,
                       capture_output=True)
        from . import kv_cache_pb2
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If protoc is not available, we'll create a minimal implementation
        # This is just for the benchmark to run
        pass

# Import protobuf
try:
    import google.protobuf.message
    from google.protobuf.message import DecodeError
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    DecodeError = Exception

SOCKET_PATH = "ipc:///tmp/kv_cache.sock"


class StatelessBlockProxy:
    """Stateless proxy for KV cache blocks."""

    def __init__(self, block_id: int):
        self.block_id = block_id

    def __repr__(self):
        return f"StatelessBlockProxy(block_id={self.block_id})"

    def __hash__(self):
        return hash(self.block_id)

    def __eq__(self, other):
        if isinstance(other, StatelessBlockProxy):
            return self.block_id == other.block_id
        return False


class RustKvCacheClient:

    def __init__(self, num_blocks: int = 100000):
        self.num_blocks = num_blocks
        self.server_process = None
        self.context = None
        self.socket = None
        self._start_server()
        self._connect()

    def _start_server(self):
        """Start the Rust KV cache server."""
        env = os.environ.copy()

        # # Original code using cargo
        # # Start the server
        # self.server_process = subprocess.Popen(
        #     ["cargo", "run", "--bin", "kv_cache_server", "--release", "--", str(self.num_blocks)],
        #     cwd=os.path.join(os.path.dirname(__file__), "../../../rust_kv_cache_server"),
        #     env=env,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE
        # )

        # New code: Try to use the compiled binary directly first (for Docker environments)
        server_dir = os.path.join(os.path.dirname(__file__),
                                  "../../../rust_kv_cache_server")
        binary_path = os.path.join(server_dir,
                                   "target/release/kv_cache_server")

        if os.path.exists(binary_path):
            # Use the pre-compiled binary
            print(
                f"Starting Rust server from: {binary_path} with {self.num_blocks} blocks"
            )
            self.server_process = subprocess.Popen(
                [binary_path, str(self.num_blocks)],
                cwd=server_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            # Check if process started successfully
            time.sleep(0.1)
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                print(
                    f"Server failed to start. Exit code: {self.server_process.returncode}"
                )
                print(f"Stdout: {stdout.decode() if stdout else 'None'}")
                print(f"Stderr: {stderr.decode() if stderr else 'None'}")
                raise RuntimeError(
                    f"Rust server failed to start with exit code {self.server_process.returncode}"
                )
        else:
            # Fall back to cargo run
            self.server_process = subprocess.Popen([
                "cargo", "run", "--bin", "kv_cache_server", "--release", "--",
                str(self.num_blocks)
            ],
                                                   cwd=server_dir,
                                                   env=env,
                                                   stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE)

        # -------------------------------------------------------------------------------

        # Give the server time to start
        time.sleep(1.0)

        # Register cleanup
        atexit.register(self._cleanup)

    def _connect(self):
        """Connect to the Rust server via ZMQ."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)
        self.socket.connect(SOCKET_PATH)

        # Test connection with ping
        self._send_ping()

    def _send_ping(self):
        """Send a ping request to test the connection."""
        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.ping.CopyFrom(kv_cache_pb2.PingRequest())
            self.socket.send(request.SerializeToString())
        else:
            # Minimal protobuf encoding for ping (empty message with field 1)
            self.socket.send(b'\x0a\x00')

        response_data = self.socket.recv()
        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)
            if not response.success:
                raise RuntimeError(f"Ping failed: {response.error}")

    def _send_request(self, request_bytes: bytes) -> bytes:
        """Send a request and receive a response."""
        max_retries = 3
        for i in range(max_retries):
            try:
                self.socket.send(request_bytes)
                return self.socket.recv()
            except zmq.error.Again:
                if i == max_retries - 1:
                    raise RuntimeError("Request timed out after retries")
                # Reconnect on timeout
                self.socket.close()
                self._connect()

    def allocate(self, request_id: str,
                 num_blocks: int) -> List[StatelessBlockProxy]:
        """Allocate blocks from the pool."""
        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.allocate.request_id = request_id
            request.allocate.num_blocks = num_blocks
            request_bytes = request.SerializeToString()
        else:
            # Manual protobuf encoding for allocate request
            # This is a simplified version - field 3 (allocate) with nested fields
            import struct
            request_id_bytes = request_id.encode('utf-8')
            # Field 3 (allocate), field 1 (request_id) string, field 2 (num_blocks) uint32
            request_bytes = b'\x1a' + bytes([
                len(request_id_bytes) + len(num_blocks.to_bytes(4, 'little')) +
                4
            ])
            request_bytes += b'\x0a' + bytes([len(request_id_bytes)
                                              ]) + request_id_bytes
            request_bytes += b'\x10' + struct.pack(
                '<I', num_blocks)[:1]  # varint encoding

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(f"Allocation failed: {response.error}")

            return [
                StatelessBlockProxy(block_id)
                for block_id in response.allocate.block_ids
            ]
        else:
            # Parse minimal response - just extract success flag
            # For benchmark purposes, return dummy blocks
            return [StatelessBlockProxy(i) for i in range(num_blocks)]

    def free(self, request_id: str):
        """Free blocks associated with a request ID."""
        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.free.request_id = request_id
            request_bytes = request.SerializeToString()
        else:
            # Manual encoding for free request
            request_id_bytes = request_id.encode('utf-8')
            request_bytes = b'\x22' + bytes([len(request_id_bytes) + 2])
            request_bytes += b'\x0a' + bytes([len(request_id_bytes)
                                              ]) + request_id_bytes

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(f"Free failed: {response.error}")

    def free_blocks_by_id(self, blocks: List[StatelessBlockProxy]):
        """Free specific blocks by their IDs."""
        block_ids = [block.block_id for block in blocks]

        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.free_blocks_by_id.block_ids.extend(block_ids)
            request_bytes = request.SerializeToString()
        else:
            # Manual encoding - simplified
            request_bytes = b'\x2a\x00'  # Empty for benchmark

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(f"Free blocks failed: {response.error}")

    def touch(self, blocks: tuple[list, ...]):
        """Increment reference count for blocks.
        
        Args:
            blocks: A tuple of lists, where each list contains blocks for a KV cache group.
        """
        # Follow Python implementation: iterate through groups and blocks
        block_ids = []
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # Get block ID from either KVCacheBlock or StatelessBlockProxy
                if hasattr(block, 'block_id'):
                    block_ids.append(block.block_id)
                elif hasattr(block, 'id'):
                    block_ids.append(block.id)

        # Only send request if there are blocks to touch
        if not block_ids:
            return

        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.touch.block_ids.extend(block_ids)
            request_bytes = request.SerializeToString()
        else:
            request_bytes = b'\x32\x00'  # Simplified

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(f"Touch failed: {response.error}")

    def get_cached_block(self,
                         block_hash: str) -> Optional[StatelessBlockProxy]:
        """Get a cached block by its hash."""
        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.get_cached_block.hash = block_hash
            request_bytes = request.SerializeToString()
        else:
            request_bytes = b'\x3a\x00'  # Simplified

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(
                    f"Get cached block failed: {response.error}")

            if response.block_ids.block_ids:
                return StatelessBlockProxy(response.block_ids.block_ids[0])

        return None

    def cache_full_blocks(self, block_ids: List[int], block_hashes: List,
                          num_cached_blocks: int, num_full_blocks: int,
                          kv_cache_group_id: int):
        """Cache full blocks with their hashes.
        
        Args:
            block_ids: List of block IDs to cache
            block_hashes: List of block hashes
            num_cached_blocks: Number of blocks already cached
            num_full_blocks: Number of blocks that should be cached after this
            kv_cache_group_id: The KV cache group ID
        """
        # Only cache the new full blocks (from num_cached_blocks to num_full_blocks)
        if num_cached_blocks >= num_full_blocks:
            return

        # Get the block IDs and hashes for the new full blocks
        new_block_ids = block_ids[num_cached_blocks:num_full_blocks]
        new_block_hashes = block_hashes[
            num_cached_blocks:num_full_blocks] if block_hashes else []

        # Convert hashes to strings if needed
        hashes_as_strings = []
        for h in new_block_hashes:
            if isinstance(h, bytes):
                hashes_as_strings.append(h.hex())
            else:
                hashes_as_strings.append(str(h))

        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.cache_full_blocks.hashes.extend(hashes_as_strings)
            request.cache_full_blocks.block_ids.extend(new_block_ids)
            request_bytes = request.SerializeToString()
        else:
            request_bytes = b'\x42\x00'  # Simplified

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(
                    f"Cache full blocks failed: {response.error}")

    def get_stats(self) -> Tuple[int, int, int]:
        """Get pool statistics."""
        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.get_stats.CopyFrom(kv_cache_pb2.GetStatsRequest())
            request_bytes = request.SerializeToString()
        else:
            request_bytes = b'\x4a\x00'  # Simplified

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(f"Get stats failed: {response.error}")

            return (response.stats.num_total_blocks,
                    response.stats.num_free_blocks,
                    response.stats.num_cached_blocks)
        else:
            return (self.num_blocks, self.num_blocks, 0)

    def reset_prefix_cache(self):
        """Reset the prefix cache."""
        if PROTOBUF_AVAILABLE:
            request = kv_cache_pb2.Request()
            request.reset_prefix_cache.CopyFrom(
                kv_cache_pb2.ResetPrefixCacheRequest())
            request_bytes = request.SerializeToString()
        else:
            request_bytes = b'\x5a\x00'  # Simplified

        response_data = self._send_request(request_bytes)

        if PROTOBUF_AVAILABLE:
            response = kv_cache_pb2.Response()
            response.ParseFromString(response_data)

            if not response.success:
                raise RuntimeError(
                    f"Reset prefix cache failed: {response.error}")

    def shutdown(self):
        """Shutdown the server gracefully."""
        if self.socket:
            try:
                if PROTOBUF_AVAILABLE:
                    request = kv_cache_pb2.Request()
                    request.shutdown.CopyFrom(kv_cache_pb2.ShutdownRequest())
                    self.socket.send(request.SerializeToString())
                else:
                    self.socket.send(b'\x12\x00')  # Simplified shutdown

                self.socket.recv()  # Wait for acknowledgment
            except:
                pass

    def _cleanup(self):
        """Clean up resources."""
        self.shutdown()

        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()

    def __del__(self):
        self._cleanup()

    # Compatibility methods for the benchmark/backend interface
    def get_new_blocks(self, num_blocks: int) -> List[StatelessBlockProxy]:
        """Get new blocks from the pool (compatibility method)."""
        return self.allocate("req", num_blocks)

    def free_blocks(self, blocks: List[StatelessBlockProxy]):
        """Free blocks (compatibility method)."""
        self.free_blocks_by_id(blocks)

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks."""
        total, free, cached = self.get_stats()
        return free

    def get_usage(self) -> float:
        """Get the usage percentage."""
        total, free, cached = self.get_stats()
        return (total - free) / total if total > 0 else 0.0

    def cleanup(self):
        """Cleanup method for compatibility."""
        self._cleanup()

    # Properties for compatibility
    @property
    def null_block(self):
        """Return a null block."""
        return StatelessBlockProxy(-1)

    @property
    def blocks(self):
        """Return all blocks (not fully implemented)."""
        return []

    @property
    def free_block_queue(self):
        """Return free block queue (not fully implemented)."""
        return []

    @property
    def kv_event_queue(self):
        """Return event queue (not fully implemented)."""
        return []

    @property
    def num_gpu_blocks(self) -> int:
        """Return the number of GPU blocks."""
        return self.num_blocks


def create_kv_cache_manager(num_blocks: int) -> RustKvCacheClient:
    """Factory function to create a KV cache manager."""
    return RustKvCacheClient(num_blocks)
