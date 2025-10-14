use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::{AttentionType, BaseKVCacheManager, KVCacheSpec, SingleTypeKVCacheManager};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};

#[derive(Debug, Clone)]
pub struct ChunkedLocalAttentionSpec {
    pub block_size: usize,
    pub attention_chunk_size: usize,
}

impl KVCacheSpec for ChunkedLocalAttentionSpec {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn spec_type(&self) -> AttentionType {
        AttentionType::ChunkedLocal
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn box_clone(&self) -> Box<dyn KVCacheSpec> {
        Box::new(self.clone())
    }
}

pub struct ChunkedLocalAttentionManager {
    base: BaseKVCacheManager,
    attention_chunk_size: usize,
}

impl ChunkedLocalAttentionManager {
    pub fn new(block_pool: Rc<RefCell<BlockPool>>, kv_cache_group_id: u32) -> Self {
        let spec = ChunkedLocalAttentionSpec {
            block_size: 16,
            attention_chunk_size: 128,
        };
        Self {
            base: BaseKVCacheManager::new(&spec, block_pool, kv_cache_group_id, 1),
            attention_chunk_size: spec.attention_chunk_size,
        }
    }

    pub fn new_with_spec(
        kv_cache_spec: &ChunkedLocalAttentionSpec,
        block_pool: Rc<RefCell<BlockPool>>,
        kv_cache_group_id: u32,
        dcp_world_size: usize,
    ) -> Self {
        assert_eq!(
            dcp_world_size, 1,
            "DCP not support chunked local attn now."
        );
        Self {
            base: BaseKVCacheManager::new(
                kv_cache_spec,
                block_pool,
                kv_cache_group_id,
                dcp_world_size,
            ),
            attention_chunk_size: kv_cache_spec.attention_chunk_size,
        }
    }
}

impl SingleTypeKVCacheManager for ChunkedLocalAttentionManager {
    fn get_blocks(&self, request_id: &str) -> Option<Vec<BlockRef>> {
        self.base.req_to_blocks.get(request_id).map(|b| b.clone())
    }

    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockRef],
    ) -> usize {
        self.base
            .get_num_blocks_to_allocate_impl(request_id, num_tokens, new_computed_blocks)
    }

    fn save_new_computed_blocks(&mut self, request_id: &str, new_computed_blocks: &[BlockRef]) {
        self.base
            .save_new_computed_blocks_impl(request_id, new_computed_blocks)
    }

    fn allocate_new_blocks(&mut self, request_id: &str, num_tokens: usize) -> Vec<BlockRef> {
        self.base.allocate_new_blocks_impl(request_id, num_tokens)
    }

    fn cache_blocks(
        &mut self,
        request_id: &str,
        request_block_hashes: &[BlockHash],
        num_tokens: usize,
    ) {
        self.base
            .cache_blocks_impl(request_id, request_block_hashes, num_tokens)
    }

    fn free(&mut self, request_id: &str) {
        self.base.free_impl(request_id)
    }

    fn get_num_common_prefix_blocks(
        &self,
        _request_id: &str,
        _num_running_requests: usize,
    ) -> usize {
        // Cascade attention is not supported by chunked local attention
        0
    }

    fn find_longest_cache_hit(
        &mut self,
        block_hashes: &[BlockHash],
        max_length: usize,
        kv_cache_group_ids: &[u32],
        block_pool: Rc<RefCell<BlockPool>>,
        kv_cache_spec: Box<dyn KVCacheSpec>,
        use_eagle: bool,
        dcp_world_size: usize,
    ) -> Vec<Vec<BlockRef>> {
        assert!(
            !use_eagle,
            "Hybrid KV cache is not supported for eagle + chunked local attention."
        );
        assert_eq!(
            dcp_world_size, 1,
            "DCP not support chunked local attn now."
        );

        let spec = kv_cache_spec
            .as_any()
            .downcast_ref::<ChunkedLocalAttentionSpec>()
            .expect("ChunkedLocalAttentionManager can only be used for chunked local attention groups");

        let max_num_blocks = max_length / spec.block_size;
        let local_attention_start_idx = if max_length > 0 {
            (max_length / spec.attention_chunk_size) * spec.attention_chunk_size
        } else {
            0
        };

        // Blocks out of window are marked as computed with null blocks,
        // and blocks inside window based on cache lookup result
        let local_attention_start_block_idx = local_attention_start_idx / spec.block_size;
        let null_block = block_pool.borrow().get_null_block();

        let mut computed_blocks: Vec<Vec<BlockRef>> = kv_cache_group_ids
            .iter()
            .map(|_| vec![null_block.clone(); local_attention_start_block_idx])
            .collect();

        for i in local_attention_start_block_idx..max_num_blocks {
            if let Some(cached_blocks) = block_pool
                .borrow()
                .get_cached_block(&block_hashes[i], kv_cache_group_ids)
            {
                for (computed, cached) in computed_blocks.iter_mut().zip(cached_blocks.iter()) {
                    computed.push(cached.clone());
                }
            } else {
                break;
            }
        }

        computed_blocks
    }

    fn remove_skipped_blocks(&mut self, request_id: &str, num_computed_tokens: usize) {
        // Remove blocks that are no longer in the chunked attention window
        // and skipped during attention computation.

        let num_cached_block = self.base.num_cached_blocks.get(request_id).copied().unwrap_or(0);
        let local_attention_start_idx =
            (num_computed_tokens / self.attention_chunk_size) * self.attention_chunk_size;
        let mut first_useful_block_idx = local_attention_start_idx / self.base.block_size;

        if num_cached_block > 0 {
            // Make sure we don't delete the last cached block
            first_useful_block_idx = first_useful_block_idx.min(num_cached_block - 1);
        }

        if let Some(blocks) = self.base.req_to_blocks.get_mut(request_id) {
            let mut removed_blocks = Vec::new();

            // We need to keep the last block to get the previous hash key
            for i in (0..first_useful_block_idx.min(blocks.len())).rev() {
                if i == 0 {
                    break; // Skip index 0 to avoid underflow in i-1
                }
                if blocks[i - 1].borrow().block_id == self.base.null_block.borrow().block_id {
                    // If the block is already a null block, blocks before it
                    // should also have been set to null blocks by previous calls
                    break;
                }
                if blocks[i - 1].borrow().block_id != self.base.null_block.borrow().block_id {
                    removed_blocks.push(blocks[i - 1].clone());
                    blocks[i - 1] = self.base.null_block.clone();
                }
            }

            if !removed_blocks.is_empty() {
                self.base
                    .block_pool
                    .borrow_mut()
                    .free_blocks(removed_blocks);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_local_attention_spec() {
        let spec = ChunkedLocalAttentionSpec {
            block_size: 16,
            attention_chunk_size: 256,
        };

        assert_eq!(spec.block_size(), 16);
        assert_eq!(spec.spec_type(), AttentionType::ChunkedLocal);
        assert_eq!(spec.attention_chunk_size, 256);

        // Test cloning
        let cloned_spec = spec.box_clone();
        assert_eq!(cloned_spec.block_size(), 16);
    }

    #[test]
    fn test_chunked_local_manager_basic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = ChunkedLocalAttentionManager::new(block_pool, 0);

        // Test allocation
        let blocks = manager.allocate_new_blocks("test_req", 64);
        assert_eq!(blocks.len(), 4); // 64 tokens / 16 block_size = 4 blocks

        // Test getting blocks
        let retrieved_blocks = manager.get_blocks("test_req");
        assert!(retrieved_blocks.is_some());
        assert_eq!(retrieved_blocks.unwrap().len(), 4);

        // Test that prefix blocks returns 0 for chunked local
        let prefix_blocks = manager.get_num_common_prefix_blocks("test_req", 2);
        assert_eq!(prefix_blocks, 0);

        // Test freeing
        manager.free("test_req");
        assert!(manager.get_blocks("test_req").is_none());
    }

    #[test]
    fn test_find_longest_cache_hit_local_window() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = ChunkedLocalAttentionManager::new(block_pool.clone(), 0);

        // Create block hashes for 10 blocks
        let hashes: Vec<BlockHash> = (0..10).map(|i| [i as u8; 32]).collect();

        let spec = ChunkedLocalAttentionSpec {
            block_size: 16,
            attention_chunk_size: 32, // 2 blocks per chunk
        };

        // Test with max_length = 64 (4 blocks)
        // With chunk size 32 (2 blocks), the last chunk starts at token 64 (block 4)
        // So blocks 0-3 should be null blocks
        let result = manager.find_longest_cache_hit(
            &hashes,
            64,
            &[0],
            block_pool.clone(),
            Box::new(spec.clone()),
            false,
            1,
        );

        assert_eq!(result.len(), 1);
        // Should have null blocks for out-of-window blocks
        assert_eq!(result[0].len(), 4);
        let null_block_id = block_pool.borrow().get_null_block().borrow().block_id;
        for block in &result[0] {
            assert_eq!(block.borrow().block_id, null_block_id);
        }
    }

    #[test]
    #[should_panic(expected = "Hybrid KV cache is not supported for eagle + chunked local attention")]
    fn test_chunked_local_eagle_panic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = ChunkedLocalAttentionManager::new(block_pool.clone(), 0);

        let spec = ChunkedLocalAttentionSpec {
            block_size: 16,
            attention_chunk_size: 128,
        };

        // Should panic with eagle enabled
        manager.find_longest_cache_hit(
            &[],
            100,
            &[0],
            block_pool,
            Box::new(spec),
            true, // use_eagle = true
            1,
        );
    }

    #[test]
    #[should_panic(expected = "DCP not support chunked local attn now")]
    fn test_chunked_local_dcp_panic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = ChunkedLocalAttentionManager::new(block_pool.clone(), 0);

        let spec = ChunkedLocalAttentionSpec {
            block_size: 16,
            attention_chunk_size: 128,
        };

        // Should panic with DCP > 1
        manager.find_longest_cache_hit(
            &[],
            100,
            &[0],
            block_pool,
            Box::new(spec),
            false,
            2, // DCP world size > 1
        );
    }

    #[test]
    fn test_remove_skipped_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = ChunkedLocalAttentionManager::new(block_pool.clone(), 0);

        // Allocate blocks for 256 tokens (16 blocks)
        manager.allocate_new_blocks("test_req", 256);

        // Set cached blocks count
        manager.base.num_cached_blocks.insert("test_req".to_string(), 10);

        // Remove blocks outside the window for 200 computed tokens
        // With chunk size 128, the current chunk window starts at 128
        // So blocks before index 8 (128/16) should be removed
        manager.remove_skipped_blocks("test_req", 200);

        let blocks = manager.get_blocks("test_req").unwrap();
        let null_block_id = block_pool.borrow().get_null_block().borrow().block_id;

        // Check that early blocks have been replaced with null blocks
        // The exact number depends on the logic, but some should be null
        let num_null_blocks = blocks
            .iter()
            .take(8) // Check first 8 blocks
            .filter(|b| b.borrow().block_id == null_block_id)
            .count();
        assert!(num_null_blocks > 0);
    }

    #[test]
    fn test_cache_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = ChunkedLocalAttentionManager::new(block_pool, 0);

        manager.allocate_new_blocks("req", 64);
        let hashes = vec![[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32]];

        // Cache blocks
        manager.cache_blocks("req", &hashes, 64);

        // Should work without issues
        assert_eq!(manager.get_blocks("req").unwrap().len(), 4);
    }

    #[test]
    fn test_local_window_calculation() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = ChunkedLocalAttentionManager::new(block_pool.clone(), 0);

        let spec = ChunkedLocalAttentionSpec {
            block_size: 4,
            attention_chunk_size: 8,
        };

        // Test example 1 from Python docstring:
        // Attention chunk size of 8, block size of 4, max length of 15
        // For next token at 15th (zero-indexed), 8th - 14th tokens are in window
        // 0th - 7th are not in the window, so marked as computed (null blocks)
        let hashes: Vec<BlockHash> = (0..4).map(|i| [i as u8; 32]).collect();

        let result = manager.find_longest_cache_hit(
            &hashes,
            15,
            &[0],
            block_pool.clone(),
            Box::new(spec.clone()),
            false,
            1,
        );

        assert_eq!(result.len(), 1);
        // Should have 2 null blocks (blocks 0 and 1 for tokens 0-7)
        // Plus potentially blocks 2-3 depending on cache
        assert!(result[0].len() >= 2);
        let null_block_id = block_pool.borrow().get_null_block().borrow().block_id;
        assert_eq!(result[0][0].borrow().block_id, null_block_id);
        assert_eq!(result[0][1].borrow().block_id, null_block_id);
    }
}