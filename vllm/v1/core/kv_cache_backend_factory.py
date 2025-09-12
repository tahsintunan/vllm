# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.request import Request

logger = init_logger(__name__)


class KVCacheBackend(ABC):
    """Abstract base class for KV cache backends."""

    @abstractmethod
    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        """Get the cached block by the block hash for each group."""
        pass

    @abstractmethod
    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching."""
        pass

    @abstractmethod
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool."""
        pass

    @abstractmethod
    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """Touch blocks to increase their reference count."""
        pass

    @abstractmethod
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks."""
        pass

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache."""
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool."""
        pass

    @abstractmethod
    def get_usage(self) -> float:
        """Get the KV cache usage."""
        pass

    @abstractmethod
    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool."""
        pass

    @property
    @abstractmethod
    def null_block(self) -> KVCacheBlock:
        """Get the null block."""
        pass

    @property
    @abstractmethod
    def blocks(self) -> list[KVCacheBlock]:
        """Get all blocks."""
        pass

    @property
    @abstractmethod
    def free_block_queue(self):
        """Get the free block queue."""
        pass

    @property
    @abstractmethod
    def cached_block_hash_to_block(self):
        """Get the cached block hash to block mapping."""
        pass

    @property
    @abstractmethod
    def kv_event_queue(self):
        """Get the KV event queue."""
        pass

    @property
    @abstractmethod
    def num_gpu_blocks(self) -> int:
        """Get the number of GPU blocks."""
        pass


class PythonBackend(KVCacheBackend):
    """Python implementation of KV cache backend using BlockPool."""

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        self.block_pool = BlockPool(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
        )

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        return self.block_pool.get_cached_block(block_hash, kv_cache_group_ids)

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        self.block_pool.cache_full_blocks(request, blocks, num_cached_blocks,
                                          num_full_blocks, block_size,
                                          kv_cache_group_id)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self.block_pool.get_new_blocks(num_blocks)

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        self.block_pool.touch(blocks)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        self.block_pool.free_blocks(ordered_blocks)

    def reset_prefix_cache(self) -> bool:
        return self.block_pool.reset_prefix_cache()

    def get_num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()

    def get_usage(self) -> float:
        return self.block_pool.get_usage()

    def take_events(self) -> list[KVCacheEvent]:
        return self.block_pool.take_events()

    @property
    def null_block(self) -> KVCacheBlock:
        return self.block_pool.null_block

    @property
    def blocks(self) -> list[KVCacheBlock]:
        return self.block_pool.blocks

    @property
    def free_block_queue(self):
        return self.block_pool.free_block_queue

    @property
    def cached_block_hash_to_block(self):
        return self.block_pool.cached_block_hash_to_block

    @property
    def kv_event_queue(self):
        return self.block_pool.kv_event_queue

    @property
    def num_gpu_blocks(self) -> int:
        return self.block_pool.num_gpu_blocks


class RustBackend(KVCacheBackend):
    """Rust backend implementation that communicates with Rust server via IPC.
    
    This backend uses a Rust process for KV cache management, communicating
    via ZeroMQ IPC sockets using Protocol Buffers serialization.
    For operations not yet implemented in Rust, it falls back to a local 
    Python BlockPool for compatibility.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        logger.info(
            "Initializing Rust KV cache backend with Protobuf IPC communication"
        )

        # Initialize the Rust client for basic operations
        from vllm.v1.core.rust_kv_cache_client import RustKvCacheClient

        self.rust_client = RustKvCacheClient(num_blocks=num_gpu_blocks)

        # Also keep a Python BlockPool for operations not yet implemented in Rust
        # This allows us to incrementally migrate functionality
        self.block_pool = BlockPool(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
        )

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        # Use Rust server for caching operations
        return self.rust_client.get_cached_block(block_hash,
                                                 kv_cache_group_ids)

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        # Use Rust server for caching operations
        # Extract block IDs and hashes for the Rust server
        block_ids = [block.block_id for block in blocks]
        block_hashes = request.block_hashes if hasattr(request,
                                                       'block_hashes') else []

        self.rust_client.cache_full_blocks(block_ids, block_hashes,
                                           num_cached_blocks, num_full_blocks,
                                           kv_cache_group_id)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        # Use Rust server for block allocation
        return self.rust_client.get_new_blocks(num_blocks)

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        # Use Rust server for touch operation
        self.rust_client.touch(blocks)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        # Use Rust server for freeing blocks
        self.rust_client.free_blocks(ordered_blocks)

    def reset_prefix_cache(self) -> bool:
        # Use Rust server for reset operation
        return self.rust_client.reset_prefix_cache()

    def get_num_free_blocks(self) -> int:
        # Use Rust server for stats
        return self.rust_client.get_num_free_blocks()

    def get_usage(self) -> float:
        # Use Rust server for stats
        return self.rust_client.get_usage()

    def take_events(self) -> list[KVCacheEvent]:
        # Use Rust client's event queue
        return self.rust_client.kv_event_queue

    @property
    def null_block(self) -> KVCacheBlock:
        # Use Rust client's null block
        return self.rust_client.null_block

    @property
    def blocks(self) -> list[KVCacheBlock]:
        # Use Rust client's blocks
        return self.rust_client.blocks

    @property
    def free_block_queue(self):
        # Use Rust client's free block queue
        return self.rust_client.free_block_queue

    @property
    def cached_block_hash_to_block(self):
        # TODO: Implement accessor in Rust client
        return self.block_pool.cached_block_hash_to_block

    @property
    def kv_event_queue(self):
        # Use Rust client's event queue
        return self.rust_client.kv_event_queue

    @property
    def num_gpu_blocks(self) -> int:
        # Use Rust client's configuration
        return self.rust_client.num_gpu_blocks


def create_kv_cache_backend(
    num_gpu_blocks: int,
    enable_caching: bool,
    enable_kv_cache_events: bool = False,
) -> KVCacheBackend:
    """Factory function to create the appropriate KV cache backend.
    
    Args:
        num_gpu_blocks: Number of GPU blocks for the cache.
        enable_caching: Whether to enable prefix caching.
        enable_kv_cache_events: Whether to enable KV cache events.
    
    Returns:
        KVCacheBackend: The selected backend implementation.
    """
    backend_type = os.environ.get('VLLM_KV_CACHE_BACKEND', 'python').lower()

    if backend_type == 'rust':
        logger.info("Using Rust KV cache backend with Protobuf")
        return RustBackend(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
        )
    else:
        if backend_type != 'python':
            logger.warning(
                f"Unknown KV cache backend '{backend_type}', falling back to Python"
            )
        logger.info("Using Python KV cache backend")
        return PythonBackend(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
        )
