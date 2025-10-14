use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::free_block_queue::{
    BlockHash, BlockHashWithGroupId, BlockRef, FreeKVCacheBlockQueue, KVCacheBlock,
};

/// Manages a fixed pool of KV cache blocks for storing attention keys/values.
/// Handles block allocation, deallocation, and prefix caching to enable memory
/// reuse across requests with common prefixes.
pub struct BlockPool {
    /// Total number of KV cache blocks available in GPU memory
    num_gpu_blocks: usize,
    /// Whether prefix caching is enabled (allows reusing blocks with matching prefixes)
    enable_caching: bool,
    /// All KV cache blocks in the pool
    blocks: Vec<BlockRef>,
    /// Queue of free blocks, maintained in LRU order for efficient allocation/eviction
    free_block_queue: FreeKVCacheBlockQueue,
    /// Maps block content `hash → {block_id → block}` for prefix cache lookups.
    /// Multiple blocks can have same content (no deduplication) to keep block tables append-only.
    cached_block_hash_to_block: HashMap<BlockHashWithGroupId, HashMap<u32, BlockRef>>,
    /// Special placeholder block `(id=0)` used to represent unallocated slots in block tables
    null_block: BlockRef,
    /// Flag for KV cache event tracking (for distributed/debugging scenarios)
    _enable_kv_cache_events: bool,
    // TODO: Add KV cache events support later
    // kv_event_queue: Vec<KVCacheEvent>,
}

impl BlockPool {
    /// Creates a new block pool with the specified number of GPU blocks.
    /// Reserves block 0 as the null block for placeholder use.
    pub fn new(num_gpu_blocks: usize, enable_caching: bool, enable_kv_cache_events: bool) -> Self {
        assert!(num_gpu_blocks > 0, "num_gpu_blocks must be positive");

        let blocks: Vec<BlockRef> = (0..num_gpu_blocks)
            .map(|idx| Rc::new(RefCell::new(KVCacheBlock::new(idx as u32))))
            .collect();

        let mut free_block_queue = FreeKVCacheBlockQueue::new(blocks.clone());

        let null_block = free_block_queue
            .popleft()
            .expect("No free blocks available");
        null_block.borrow_mut().is_null = true;

        Self {
            num_gpu_blocks,
            enable_caching,
            blocks,
            free_block_queue,
            cached_block_hash_to_block: HashMap::new(),
            null_block,
            _enable_kv_cache_events: enable_kv_cache_events,
        }
    }

    /// Retrieves cached blocks by their content hash for each KV cache group.
    /// Returns None if any group has a cache miss.
    pub fn get_cached_block(
        &self,
        block_hash: &BlockHash,
        kv_cache_group_ids: &[u32],
    ) -> Option<Vec<BlockRef>> {
        let mut cached_blocks = Vec::with_capacity(kv_cache_group_ids.len());

        for &group_id in kv_cache_group_ids {
            let hash_key = BlockHashWithGroupId {
                hash: *block_hash,
                group_id,
            };

            let cached_blocks_one_group = self.cached_block_hash_to_block.get(&hash_key)?;

            if cached_blocks_one_group.is_empty() {
                return None;
            }

            let first_block = cached_blocks_one_group.values().next()?.clone();
            cached_blocks.push(first_block);
        }

        Some(cached_blocks)
    }

    /// Marks full blocks as cached by updating their hash metadata.
    /// Only caches newly full blocks between num_cached_blocks and num_full_blocks.
    pub fn cache_full_blocks(
        &mut self,
        request_block_hashes: &[BlockHash],
        blocks: &[BlockRef],
        num_cached_blocks: usize,
        num_full_blocks: usize,
        _block_size: usize, // TODO: use when kv_event_queue is implemented
        kv_cache_group_id: u32,
    ) {
        if num_cached_blocks == num_full_blocks {
            return;
        }

        let new_full_blocks = &blocks[num_cached_blocks..num_full_blocks];
        assert!(
            request_block_hashes.len() >= num_full_blocks,
            "Insufficient block hashes: got {}, need at least {}",
            request_block_hashes.len(),
            num_full_blocks
        );
        let new_block_hashes = &request_block_hashes[num_cached_blocks..];

        for (i, block) in new_full_blocks.iter().enumerate() {
            let mut block_mut = block.borrow_mut();
            assert!(
                block_mut.hash_key.is_none(),
                "Block {} already cached - cannot cache twice",
                block_mut.block_id
            );

            let block_hash = new_block_hashes[i];
            let hash_key = BlockHashWithGroupId {
                hash: block_hash,
                group_id: kv_cache_group_id,
            };

            block_mut.hash_key = Some(hash_key);
            let block_id = block_mut.block_id;
            drop(block_mut); // Release borrow before storing in HashMap

            self.cached_block_hash_to_block
                .entry(hash_key)
                .or_insert_with(HashMap::new)
                .insert(block_id, block.clone());
        }

        // TODO: Handle KV cache events if needed
    }

    /// Allocates new blocks from the free pool, evicting cached blocks if needed.
    /// Panics if insufficient blocks are available.
    pub fn get_new_blocks(&mut self, num_blocks: usize) -> Vec<BlockRef> {
        assert!(
            num_blocks <= self.get_num_free_blocks(),
            "Cannot get {} free blocks from the pool, only {} available",
            num_blocks,
            self.get_num_free_blocks()
        );

        let new_blocks = self.free_block_queue.popleft_n(num_blocks);

        if self.enable_caching {
            for block in &new_blocks {
                self.maybe_evict_cached_block(block);
                let mut block_mut = block.borrow_mut();
                assert_eq!(
                    block_mut.ref_cnt, 0,
                    "Block {} has ref_cnt {} but should be 0 when in free queue",
                    block_mut.block_id, block_mut.ref_cnt
                );
                block_mut.ref_cnt += 1;
            }
        } else {
            for block in &new_blocks {
                let mut block_mut = block.borrow_mut();
                assert_eq!(
                    block_mut.ref_cnt, 0,
                    "Block {} has ref_cnt {} but should be 0 when in free queue",
                    block_mut.block_id, block_mut.ref_cnt
                );
                block_mut.ref_cnt += 1;
            }
        }

        new_blocks
    }

    /// Evicts a block from the cache if it was previously cached.
    /// Returns true if eviction occurred.
    fn maybe_evict_cached_block(&mut self, block: &BlockRef) -> bool {
        let hash_key = {
            let block_borrow = block.borrow();
            block_borrow.hash_key
        };

        if let Some(hash_key) = hash_key {
            if let Some(blocks_by_id) = self.cached_block_hash_to_block.get_mut(&hash_key) {
                let block_id = block.borrow().block_id;
                block.borrow_mut().hash_key = None; // reset hash key

                blocks_by_id.remove(&block_id);
                if blocks_by_id.is_empty() {
                    self.cached_block_hash_to_block.remove(&hash_key);
                }

                // TODO: Handle KV cache events if needed
                return true;
            }
        }

        false
    }

    /// Increments reference counts for blocks, removing them from free queue if needed.
    /// Used when cached blocks are reused by new requests.
    pub fn touch(&mut self, blocks: &[Vec<BlockRef>]) {
        for blocks_per_group in blocks {
            for block in blocks_per_group {
                let mut block_mut = block.borrow_mut();
                // ref_cnt=0 means this block is in the free list (eviction candidate)
                if block_mut.ref_cnt == 0 && !block_mut.is_null {
                    drop(block_mut); // Release borrow before modifying queue
                    self.free_block_queue.remove(block);
                    block.borrow_mut().ref_cnt += 1;
                } else {
                    block_mut.ref_cnt += 1;
                }
            }
        }
    }

    /// Decrements reference counts and returns blocks to free queue when ref_cnt reaches 0.
    /// Blocks should be ordered by eviction priority.
    pub fn free_blocks(&mut self, ordered_blocks: Vec<BlockRef>) {
        // First pass: decrement ref counts
        for block in &ordered_blocks {
            let mut b = block.borrow_mut();

            // Python doesn't check for double-free, which can cause duplicates in free queue.
            // TODO: Consider using debug_assert! for performance.
            assert!(
                b.ref_cnt > 0 || b.is_null,
                "Block {} double-freed: ref_cnt already 0",
                b.block_id
            );

            // Don't modify ref_cnt for null blocks
            if !b.is_null {
                b.ref_cnt -= 1;
            }
            // b.ref_cnt = b.ref_cnt.saturating_sub(1); // underflow protection
        }

        // Second pass: append to free queue if ref_cnt == 0
        let blocks_to_free: Vec<BlockRef> = ordered_blocks
            .into_iter()
            .filter(|block| {
                let b = block.borrow();
                b.ref_cnt == 0 && !b.is_null
            })
            .collect();

        self.free_block_queue.append_n(&blocks_to_free);
    }

    /// Clears all cached blocks and hash mappings. Used after model weight updates in RLHF.
    /// Returns false if blocks are still in use.
    pub fn reset_prefix_cache(&mut self) -> bool {
        let num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks();
        if num_used_blocks != 1 {
            // The null block is always marked as used
            // Failed to reset because blocks are still in use
            // TODO: Add logging when logger is implemented
            // log::warn!("Failed to reset prefix cache: {} blocks still in use", num_used_blocks - 1);
            return false;
        }

        // Clear all cached blocks
        self.cached_block_hash_to_block.clear();

        // Reset hash for all blocks
        for block in &self.blocks {
            block.borrow_mut().hash_key = None;
        }

        // TODO: Handle KV cache events if needed

        true
    }

    /// Returns KV cache usage as a fraction between 0.0 and 1.0.
    pub fn get_usage(&self) -> f32 {
        // Subtract 1 to account for null block
        let total_gpu_blocks = self.num_gpu_blocks - 1;
        if total_gpu_blocks == 0 {
            return 0.0;
        }
        1.0 - (self.get_num_free_blocks() as f32 / total_gpu_blocks as f32)
    }

    /// Returns the number of blocks available for allocation.
    #[inline]
    pub fn get_num_free_blocks(&self) -> usize {
        self.free_block_queue.len()
    }

    /// Returns the special null block used for placeholder slots.
    #[inline]
    pub fn get_null_block(&self) -> BlockRef {
        self.null_block.clone()
    }

    // TODO: Add take_events() when we implement KV cache events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_pool_creation() {
        let pool = BlockPool::new(10, true, false);
        assert_eq!(pool.get_num_free_blocks(), 9); // 10 - 1 null block
        assert!(pool.get_usage() < 0.01); // Should be close to 0
    }

    #[test]
    fn test_get_new_blocks() {
        let mut pool = BlockPool::new(10, false, false);
        let blocks = pool.get_new_blocks(3);
        assert_eq!(blocks.len(), 3);
        assert_eq!(pool.get_num_free_blocks(), 6); // 9 - 3

        // Check ref counts
        for block in blocks {
            assert_eq!(block.borrow().ref_cnt, 1);
        }
    }

    #[test]
    fn test_free_blocks() {
        let mut pool = BlockPool::new(10, false, false);
        let blocks = pool.get_new_blocks(3);
        let initial_free = pool.get_num_free_blocks();

        pool.free_blocks(blocks);
        assert_eq!(pool.get_num_free_blocks(), initial_free + 3);
    }

    #[test]
    fn test_touch_blocks() {
        let mut pool = BlockPool::new(10, false, false);
        let blocks = pool.get_new_blocks(2);

        // Free blocks first
        pool.free_blocks(blocks.clone());

        // Touch should increment ref count and remove from free queue
        pool.touch(&vec![blocks.clone()]);

        for block in blocks {
            assert_eq!(block.borrow().ref_cnt, 1);
        }
    }

    #[test]
    fn test_null_block_handling() {
        let pool = BlockPool::new(5, false, false);
        let null_block = pool.get_null_block();

        // Null block should have special properties
        assert!(null_block.borrow().is_null);
        assert_eq!(null_block.borrow().block_id, 0);

        // Should have 4 free blocks (5 - 1 null block)
        assert_eq!(pool.get_num_free_blocks(), 4);
    }

    #[test]
    fn test_free_queue_integration() {
        let mut pool = BlockPool::new(10, false, false);

        // Get some blocks
        let blocks1 = pool.get_new_blocks(3);
        let blocks2 = pool.get_new_blocks(2);

        assert_eq!(pool.get_num_free_blocks(), 4); // 9 - 3 - 2

        // Free blocks in specific order (tests eviction priority)
        pool.free_blocks(blocks2.clone());
        pool.free_blocks(blocks1.clone());

        assert_eq!(pool.get_num_free_blocks(), 9);

        // Get all remaining blocks to test the queue order
        let new_blocks = pool.get_new_blocks(9);
        assert_eq!(new_blocks.len(), 9);

        // The queue is FIFO: remaining blocks (6-9) come first,
        // then blocks2 (4-5), then blocks1 (1-3)
        assert_eq!(new_blocks[0].borrow().block_id, 6);
        assert_eq!(new_blocks[1].borrow().block_id, 7);
        assert_eq!(new_blocks[2].borrow().block_id, 8);
        assert_eq!(new_blocks[3].borrow().block_id, 9);
        assert_eq!(
            new_blocks[4].borrow().block_id,
            blocks2[0].borrow().block_id
        ); // 4
        assert_eq!(
            new_blocks[5].borrow().block_id,
            blocks2[1].borrow().block_id
        ); // 5
        assert_eq!(
            new_blocks[6].borrow().block_id,
            blocks1[0].borrow().block_id
        ); // 1
        assert_eq!(
            new_blocks[7].borrow().block_id,
            blocks1[1].borrow().block_id
        ); // 2
        assert_eq!(
            new_blocks[8].borrow().block_id,
            blocks1[2].borrow().block_id
        ); // 3
    }

    #[test]
    fn test_ref_counting_with_touch() {
        let mut pool = BlockPool::new(5, false, false);
        let blocks = pool.get_new_blocks(2);

        // Initial ref_cnt should be 1
        assert_eq!(blocks[0].borrow().ref_cnt, 1);

        // Touch increments ref_cnt
        pool.touch(&vec![blocks.clone()]);
        assert_eq!(blocks[0].borrow().ref_cnt, 2);

        // Free once - should not return to free queue
        pool.free_blocks(blocks.clone());
        assert_eq!(blocks[0].borrow().ref_cnt, 1);
        assert_eq!(pool.get_num_free_blocks(), 2); // Still 2 free

        // Free again - should return to free queue
        pool.free_blocks(blocks.clone());
        assert_eq!(blocks[0].borrow().ref_cnt, 0);
        assert_eq!(pool.get_num_free_blocks(), 4); // Now 4 free
    }

    #[test]
    fn test_cache_operations() {
        let mut pool = BlockPool::new(10, true, false);
        let blocks = pool.get_new_blocks(3);

        // Create mock block hashes
        let block_hash1 = [1u8; 32];
        let block_hash2 = [2u8; 32];
        let block_hash3 = [3u8; 32];
        let block_hashes = vec![block_hash1, block_hash2, block_hash3];

        // Cache the blocks
        pool.cache_full_blocks(&block_hashes, &blocks, 0, 3, 16, 0);

        // Blocks should now have hashes
        assert!(blocks[0].borrow().hash_key.is_some());
        assert!(blocks[1].borrow().hash_key.is_some());
        assert!(blocks[2].borrow().hash_key.is_some());

        // Should be able to retrieve cached blocks
        let cached = pool.get_cached_block(&block_hash1, &[0]);
        assert!(cached.is_some());
        assert_eq!(
            cached.unwrap()[0].borrow().block_id,
            blocks[0].borrow().block_id
        );

        // Cache miss for non-existent hash
        let block_hash_miss = [99u8; 32];
        let cached_miss = pool.get_cached_block(&block_hash_miss, &[0]);
        assert!(cached_miss.is_none());
    }

    #[test]
    fn test_eviction_clears_cache() {
        let mut pool = BlockPool::new(5, true, false);
        let blocks = pool.get_new_blocks(2);

        // Cache a block
        let block_hash = [1u8; 32];
        pool.cache_full_blocks(&vec![block_hash], &blocks[0..1], 0, 1, 16, 0);

        // Free the block back to pool
        pool.free_blocks(vec![blocks[0].clone()]);

        // Get a new block - should trigger eviction if it reuses the same block
        let new_blocks = pool.get_new_blocks(1);

        if new_blocks[0].borrow().block_id == blocks[0].borrow().block_id {
            // The block was reused, hash should be cleared
            assert!(new_blocks[0].borrow().hash_key.is_none());
        }
    }

    #[test]
    #[should_panic(expected = "Cannot get 10 free blocks from the pool")]
    fn test_allocation_exceeds_capacity() {
        let mut pool = BlockPool::new(5, false, false);
        pool.get_new_blocks(10); // Should panic - only 4 free blocks available
    }

    #[test]
    fn test_reset_prefix_cache() {
        let mut pool = BlockPool::new(5, true, false);
        let blocks = pool.get_new_blocks(2);

        // Cache blocks
        let block_hashes = vec![[1u8; 32], [2u8; 32]];
        pool.cache_full_blocks(&block_hashes, &blocks, 0, 2, 16, 0);

        // Can't reset while blocks are in use
        assert!(!pool.reset_prefix_cache());

        // Free all blocks
        pool.free_blocks(blocks);

        // Now reset should work
        assert!(pool.reset_prefix_cache());

        // Cache should be cleared
        let cached = pool.get_cached_block(&[1u8; 32], &[0]);
        assert!(cached.is_none());
    }

    #[test]
    #[should_panic(expected = "Block 1 double-freed: ref_cnt already 0")]
    fn test_ref_count_underflow_protection() {
        let mut pool = BlockPool::new(5, false, false);
        let blocks = pool.get_new_blocks(2);

        // Initial ref_cnt should be 1
        assert_eq!(blocks[0].borrow().ref_cnt, 1);
        assert_eq!(blocks[1].borrow().ref_cnt, 1);

        // Free once - ref_cnt should go to 0
        pool.free_blocks(blocks.clone());
        assert_eq!(blocks[0].borrow().ref_cnt, 0);
        assert_eq!(blocks[1].borrow().ref_cnt, 0);
        assert_eq!(pool.get_num_free_blocks(), 4); // 2 blocks returned to pool

        // Free again - should panic on double-free
        pool.free_blocks(blocks.clone()); // This will panic
    }

    #[test]
    #[should_panic(expected = "Block 1 already cached - cannot cache twice")]
    fn test_panic_on_double_cache() {
        let mut pool = BlockPool::new(10, true, false);
        let blocks = pool.get_new_blocks(1);
        let hash = [1u8; 32];

        // Cache the block once
        pool.cache_full_blocks(&vec![hash], &blocks, 0, 1, 16, 0);

        // Verify it was cached
        assert!(blocks[0].borrow().hash_key.is_some());

        // Try to cache the same block again - should panic
        pool.cache_full_blocks(&vec![hash], &blocks, 0, 1, 16, 0);
    }
}
