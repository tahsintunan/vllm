# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Union

import zmq

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import (
    KVCacheManager as PythonKVCacheManager)
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RustKVCacheManager:
    """Python client for Rust KV cache manager.

    This is a thin client that communicates with a Rust process
    for KV cache management via ZMQ. It maintains the same
    interface as the Python KVCacheManager but delegates all
    operations to the Rust backend.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = False,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
    ):
        """Initialize the Rust KV cache manager client.

        Args:
            kv_cache_config: Configuration for KV cache.
            max_model_len: Maximum model sequence length.
            enable_caching: Whether to enable prefix caching.
            use_eagle: Whether Eagle speculation is enabled.
            log_stats: Whether to log statistics.
            enable_kv_cache_events: Whether to enable KV cache events.
            dcp_world_size: World size for distributed checkpointing.
        """
        logger.info("Initializing Rust KV cache manager client")

        # Store configuration
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.enable_kv_cache_events = enable_kv_cache_events
        self.dcp_world_size = dcp_world_size

        # Start the Rust process
        self._start_rust_process()

        # Setup ZMQ connection
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.zmq_port}")

        # Set timeout to avoid hanging
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout

        # Register cleanup
        atexit.register(self._cleanup)

        # Send initialization message
        self._send_init_message()

    def _start_rust_process(self):
        """Start the Rust KV cache manager process."""
        # Find the Rust binary
        rust_binary = os.environ.get(
            "RUST_KV_CACHE_BINARY",
            str(
                Path(__file__).parent.parent.parent / "v1" / "rust_core" /
                "target" / "release" / "vllm-core-server"))

        if not os.path.exists(rust_binary):
            raise FileNotFoundError(
                f"Rust KV cache binary not found at {rust_binary}. "
                "Please build it with: cargo build --release --bin vllm-core-server"
            )

        # Get available port for ZMQ
        self.zmq_port = int(os.environ.get("RUST_KV_ZMQ_PORT", "5555"))

        # Get number of blocks from config
        num_blocks = getattr(self.kv_cache_config, 'num_blocks', 10000)

        # Start the process
        env = os.environ.copy()
        env["RUST_LOG"] = os.environ.get("RUST_LOG", "info")
        env["ENABLE_PREFIX_CACHING"] = str(self.enable_caching).lower()
        env["ZMQ_PORT"] = str(self.zmq_port)

        logger.info(f"Starting Rust KV cache process: {rust_binary}")
        logger.info(f"  ZMQ port: {self.zmq_port}")
        logger.info(f"  Num blocks: {num_blocks}")
        logger.info(f"  Enable caching: {self.enable_caching}")

        self.rust_process = subprocess.Popen(
            [rust_binary, str(num_blocks)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for process to start
        time.sleep(0.5)

        # Check if process is running
        if self.rust_process.poll() is not None:
            stdout, stderr = self.rust_process.communicate()
            raise RuntimeError(f"Rust KV cache process failed to start:\n"
                               f"stdout: {stdout.decode()}\n"
                               f"stderr: {stderr.decode()}")

        logger.info(
            f"Rust KV cache process started with PID {self.rust_process.pid}")

    def _send_init_message(self):
        """Send initialization message to Rust process."""
        init_msg = {
            "method": "initialize",
            "params": {
                "max_model_len": self.max_model_len,
                "enable_caching": self.enable_caching,
                "use_eagle": self.use_eagle,
                "log_stats": self.log_stats,
                "enable_kv_cache_events": self.enable_kv_cache_events,
                "dcp_world_size": self.dcp_world_size,
                "num_blocks": getattr(self.kv_cache_config, 'num_blocks',
                                      10000),
            }
        }

        response = self._send_request(init_msg)
        if not response.get("success", False):
            raise RuntimeError(
                f"Failed to initialize Rust KV cache: {response.get('error', 'Unknown error')}"
            )

        logger.info("Successfully initialized Rust KV cache manager")

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the Rust process and get response."""
        try:
            # Send request as JSON
            self.socket.send_json(request)

            # Receive response
            response = self.socket.recv_json()
            return response

        except zmq.Again:
            logger.error(
                f"ZMQ timeout while sending request: {request.get('method', 'unknown')}"
            )
            raise TimeoutError("Request to Rust KV cache timed out")
        except Exception as e:
            logger.error(f"Error communicating with Rust KV cache: {e}")
            raise

    def _cleanup(self):
        """Cleanup resources on shutdown."""
        try:
            # Send shutdown message
            if hasattr(self, 'socket'):
                try:
                    self.socket.send_json({"method": "shutdown", "params": {}})
                    self.socket.recv_json()
                except:
                    pass

                self.socket.close()
                self.context.term()

            # Terminate Rust process
            if hasattr(self, 'rust_process'):
                self.rust_process.terminate()
                try:
                    self.rust_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.rust_process.kill()
                    self.rust_process.wait()

                logger.info("Rust KV cache process terminated")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def allocate_slots(self,
                       request,
                       num_computed_tokens: int,
                       num_new_tokens: int,
                       new_spec_token_ids=None):
        """Allocate KV cache slots for the request."""
        msg = {
            "method": "allocate_slots",
            "params": {
                "request_id": request.request_id,
                "prompt_token_ids": request.inputs.prompt_token_ids,
                "num_computed_tokens": num_computed_tokens,
                "num_new_tokens": num_new_tokens,
                "spec_token_ids": new_spec_token_ids or [],
            }
        }

        response = self._send_request(msg)
        if not response.get("success", False):
            raise RuntimeError(
                f"Failed to allocate slots: {response.get('error', 'Unknown error')}"
            )

        # Convert response to KVCacheBlocks format
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks
        block_ids = response.get("block_ids", [])
        # TODO: Handle multi-group blocks properly
        return KVCacheBlocks([[bid] for bid in block_ids])

    def free(self, request) -> None:
        """Free all KV cache blocks for the request."""
        msg = {
            "method": "free",
            "params": {
                "request_id": request.request_id,
            }
        }

        response = self._send_request(msg)
        if not response.get("success", False):
            logger.warning(
                f"Failed to free blocks for request {request.request_id}: {response.get('error')}"
            )

    def get_computed_blocks(self, request):
        """Get computed blocks for the request."""
        msg = {
            "method": "get_computed_blocks",
            "params": {
                "request_id": request.request_id,
                "prompt_token_ids": request.inputs.prompt_token_ids,
            }
        }

        response = self._send_request(msg)
        if not response.get("success", False):
            # Return empty if failed
            return ([], 0)

        computed_blocks = response.get("computed_blocks", [])
        num_computed_tokens = response.get("num_computed_tokens", 0)
        return (computed_blocks, num_computed_tokens)

    def create_empty_block_list(self):
        """Create an empty block list."""
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        # Get number of KV cache groups from config
        num_groups = len(self.kv_cache_config.kv_cache_groups) if hasattr(
            self.kv_cache_config, 'kv_cache_groups') else 1
        return KVCacheBlocks.new_empty(num_groups)

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks."""
        msg = {"method": "get_num_free_blocks", "params": {}}

        response = self._send_request(msg)
        return response.get("num_free_blocks", 0)

    def get_usage(self) -> float:
        """Get cache usage percentage."""
        msg = {"method": "get_usage", "params": {}}

        response = self._send_request(msg)
        return response.get("usage", 0.0)

    def take_events(self):
        """Take pending KV cache events."""
        msg = {"method": "take_events", "params": {}}

        response = self._send_request(msg)
        return response.get("events", [])

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache."""
        msg = {"method": "reset_prefix_cache", "params": {}}

        response = self._send_request(msg)
        return response.get("success", False)

    def get_blocks(self, request_id: str):
        """Get blocks for a request."""
        msg = {
            "method": "get_blocks",
            "params": {
                "request_id": request_id,
            }
        }

        response = self._send_request(msg)
        if not response.get("success", False):
            from vllm.v1.core.kv_cache_manager import KVCacheBlocks
            return KVCacheBlocks.new_empty(1)

        from vllm.v1.core.kv_cache_manager import KVCacheBlocks
        block_ids = response.get("block_ids", [])
        return KVCacheBlocks([[bid] for bid in block_ids])

    def get_block_ids(self, request_id: str):
        """Get block IDs for a request."""
        msg = {
            "method": "get_block_ids",
            "params": {
                "request_id": request_id,
            }
        }

        response = self._send_request(msg)
        if not response.get("success", False):
            return ([], )

        block_ids = response.get("block_ids", [])
        return (block_ids, )

    def cache_blocks(self, request, num_computed_tokens: int) -> None:
        """Cache computed blocks."""
        msg = {
            "method": "cache_blocks",
            "params": {
                "request_id": request.request_id,
                "prompt_token_ids": request.inputs.prompt_token_ids,
                "num_computed_tokens": num_computed_tokens,
            }
        }

        response = self._send_request(msg)
        if not response.get("success", False):
            logger.warning(f"Failed to cache blocks: {response.get('error')}")

    def get_num_common_prefix_blocks(self, request1, request2) -> int:
        """Get number of common prefix blocks between two requests."""
        msg = {
            "method": "get_num_common_prefix_blocks",
            "params": {
                "request1_id": request1.request_id,
                "request1_prompt": request1.inputs.prompt_token_ids,
                "request2_id": request2.request_id,
                "request2_prompt": request2.inputs.prompt_token_ids,
            }
        }

        response = self._send_request(msg)
        return response.get("num_common_blocks", 0)

    def make_prefix_cache_stats(self):
        """Get prefix cache statistics."""
        msg = {"method": "get_prefix_cache_stats", "params": {}}

        response = self._send_request(msg)
        if not response.get("has_stats", False):
            return None

        stats = response.get("stats", {})
        # Convert to PrefixCacheStats format
        from vllm.v1.core.prefix_caching import PrefixCacheStats
        return PrefixCacheStats(
            num_cached_tokens=stats.get("num_cached_tokens", 0),
            num_queries=stats.get("num_queries", 0),
            num_hits=stats.get("num_hits", 0),
        )

    @property
    def usage(self) -> float:
        """Get cache usage as a property."""
        return self.get_usage()


def create_kv_cache_manager(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    enable_caching: bool = False,
    use_eagle: bool = False,
    log_stats: bool = False,
    enable_kv_cache_events: bool = False,
    dcp_world_size: int = 1,
) -> Union[PythonKVCacheManager, RustKVCacheManager]:
    """Factory function to create the appropriate KV cache manager.

    Creates either a Python KVCacheManager or a Rust client based on
    the USE_RUST_KV environment variable.

    Args:
        kv_cache_config: Configuration for KV cache.
        max_model_len: Maximum model sequence length.
        enable_caching: Whether to enable prefix caching.
        use_eagle: Whether Eagle speculation is enabled.
        log_stats: Whether to log statistics.
        enable_kv_cache_events: Whether to enable KV cache events.
        dcp_world_size: World size for distributed checkpointing.

    Returns:
        Either KVCacheManager (Python) or RustKVCacheManager (Rust client).
    """
    use_rust = os.environ.get('USE_RUST_KV', '0') == '1'

    if use_rust:
        logger.info("Creating Rust KV cache manager client")
        return RustKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
    else:
        logger.info("Creating Python KV cache manager")
        return PythonKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
