use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::{cdiv, AttentionType, BaseKVCacheManager, KVCacheSpec, SingleTypeKVCacheManager};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};

#[derive(Debug, Clone)]
pub struct SlidingWindowSpec {
    pub block_size: usize,
    pub sliding_window: usize,
}

impl KVCacheSpec for SlidingWindowSpec {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn spec_type(&self) -> AttentionType {
        AttentionType::SlidingWindow
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn box_clone(&self) -> Box<dyn KVCacheSpec> {
        Box::new(self.clone())
    }
}

pub struct SlidingWindowManager {
    base: BaseKVCacheManager,
    sliding_window: usize,
}

impl SlidingWindowManager {
    pub fn new(block_pool: Rc<RefCell<BlockPool>>, kv_cache_group_id: u32) -> Self {
        // Use a default spec for simple construction
        let spec = SlidingWindowSpec {
            block_size: 16,
            sliding_window: 128,
        };
        Self {
            base: BaseKVCacheManager::new(&spec, block_pool, kv_cache_group_id, 1),
            sliding_window: spec.sliding_window,
        }
    }

    pub fn new_with_spec(
        kv_cache_spec: &SlidingWindowSpec,
        block_pool: Rc<RefCell<BlockPool>>,
        kv_cache_group_id: u32,
        dcp_world_size: usize,
    ) -> Self {
        Self {
            base: BaseKVCacheManager::new(
                kv_cache_spec,
                block_pool,
                kv_cache_group_id,
                dcp_world_size,
            ),
            sliding_window: kv_cache_spec.sliding_window,
        }
    }
}

impl SingleTypeKVCacheManager for SlidingWindowManager {
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
        // Prefix blocks are null blocks for sliding window layers
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
        assert_eq!(
            dcp_world_size, 1,
            "DCP not supported for sliding window attention"
        );

        let block_size = kv_cache_spec.block_size();
        let sliding_window =
            if let Some(spec) = kv_cache_spec.as_any().downcast_ref::<SlidingWindowSpec>() {
                spec.sliding_window
            } else {
                panic!("Invalid spec type for SlidingWindowManager");
            };

        let mut sliding_window_contiguous_blocks = cdiv(sliding_window - 1, block_size);
        if use_eagle {
            sliding_window_contiguous_blocks += 1;
        }

        let max_num_blocks = max_length / block_size;
        let null_block = block_pool.borrow().get_null_block();
        let mut computed_blocks: Vec<Vec<BlockRef>> =
            vec![vec![null_block.clone(); max_num_blocks]; kv_cache_group_ids.len()];

        let mut num_contiguous_blocks = 0;
        let mut match_found = false;

        // Search from right to left
        for i in (0..max_num_blocks).rev() {
            if let Some(cached_blocks) = block_pool
                .borrow()
                .get_cached_block(&block_hashes[i], kv_cache_group_ids)
            {
                for (computed, cached) in computed_blocks.iter_mut().zip(cached_blocks.iter()) {
                    computed[i] = cached.clone();
                }
                num_contiguous_blocks += 1;
                if num_contiguous_blocks >= sliding_window_contiguous_blocks {
                    // Trim trailing blocks
                    for computed in &mut computed_blocks {
                        computed.truncate(i + num_contiguous_blocks);
                    }
                    match_found = true;
                    break;
                }
            } else {
                num_contiguous_blocks = 0;
            }
        }

        if !match_found {
            for computed in &mut computed_blocks {
                computed.truncate(num_contiguous_blocks);
            }
        }

        if use_eagle && !computed_blocks[0].is_empty() {
            for computed in &mut computed_blocks {
                computed.pop();
            }
        }

        computed_blocks
    }

    fn remove_skipped_blocks(&mut self, request_id: &str, num_computed_tokens: usize) {
        let last_useful_token = num_computed_tokens.saturating_sub(self.sliding_window - 1);
        let last_useful_block = last_useful_token / self.base.block_size;

        if let Some(blocks) = self.base.req_to_blocks.get_mut(request_id) {
            let mut removed_blocks = Vec::new();

            for i in (0..last_useful_block.min(blocks.len())).rev() {
                if blocks[i].borrow().block_id == self.base.null_block.borrow().block_id {
                    break;
                }
                removed_blocks.push(blocks[i].clone());
                blocks[i] = self.base.null_block.clone();
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
    fn test_sliding_window_manager_basic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = SlidingWindowManager::new(block_pool, 0);

        // Test allocation
        let blocks = manager.allocate_new_blocks("test_req", 64);
        assert_eq!(blocks.len(), 4); // 64 tokens / 16 block_size = 4 blocks

        // Test getting blocks
        let retrieved_blocks = manager.get_blocks("test_req");
        assert!(retrieved_blocks.is_some());
        assert_eq!(retrieved_blocks.unwrap().len(), 4);

        // Test that prefix blocks returns 0 for sliding window
        let prefix_blocks = manager.get_num_common_prefix_blocks("test_req", 2);
        assert_eq!(prefix_blocks, 0);

        // Test freeing
        manager.free("test_req");
        assert!(manager.get_blocks("test_req").is_none());
    }

    #[test]
    fn test_sliding_window_spec() {
        let spec = SlidingWindowSpec {
            block_size: 16,
            sliding_window: 256,
        };

        assert_eq!(spec.block_size(), 16);
        assert_eq!(spec.spec_type(), AttentionType::SlidingWindow);
        assert_eq!(spec.sliding_window, 256);

        // Test cloning
        let cloned_spec = spec.box_clone();
        assert_eq!(cloned_spec.block_size(), 16);
    }

    #[test]
    fn test_remove_skipped_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = SlidingWindowManager::new(block_pool.clone(), 0);

        // Allocate some blocks
        manager.allocate_new_blocks("test_req", 256);

        // Remove blocks outside sliding window
        manager.remove_skipped_blocks("test_req", 200);

        // Check that some blocks have been replaced with null blocks
        let blocks = manager.get_blocks("test_req").unwrap();
        let null_block_id = block_pool.borrow().get_null_block().borrow().block_id;

        // Early blocks should be null blocks
        let num_null_blocks = blocks
            .iter()
            .filter(|b| b.borrow().block_id == null_block_id)
            .count();
        assert!(num_null_blocks > 0);
    }

    #[test]
    fn test_remove_skipped_blocks_edge_cases() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = SlidingWindowManager::new(block_pool.clone(), 0);

        // Test with num_computed_tokens < sliding_window (no blocks should be removed)
        manager.allocate_new_blocks("req1", 64);
        let blocks_before = manager.get_blocks("req1").unwrap().len();
        manager.remove_skipped_blocks("req1", 50); // 50 < 128 (sliding_window)
        let blocks_after = manager.get_blocks("req1").unwrap();
        assert_eq!(blocks_after.len(), blocks_before);

        // No blocks should be null since we're within sliding window
        let null_block_id = block_pool.borrow().get_null_block().borrow().block_id;
        let null_count = blocks_after
            .iter()
            .filter(|b| b.borrow().block_id == null_block_id)
            .count();
        assert_eq!(null_count, 0);

        // Test with empty request
        manager.remove_skipped_blocks("nonexistent", 200);
        assert!(manager.get_blocks("nonexistent").is_none());
    }

    #[test]
    fn test_save_new_computed_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = SlidingWindowManager::new(block_pool.clone(), 0);

        // Create computed blocks
        let blocks: Vec<_> = (0..3)
            .map(|_| block_pool.borrow_mut().get_new_blocks(1)[0].clone())
            .collect();

        // Save for new request
        manager.save_new_computed_blocks("new_req", &blocks);
        assert_eq!(manager.get_blocks("new_req").unwrap().len(), 3);

        // Save empty for existing (should be fine)
        manager.save_new_computed_blocks("new_req", &[]);
        assert_eq!(manager.get_blocks("new_req").unwrap().len(), 3);
    }

    #[test]
    fn test_cache_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = SlidingWindowManager::new(block_pool, 0);

        manager.allocate_new_blocks("req", 64);
        let hashes = vec![[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32]];

        // Cache partial blocks
        manager.cache_blocks("req", &hashes, 48); // 3 full blocks (48/16 = 3)

        // Cache all blocks
        manager.cache_blocks("req", &hashes, 64); // 4 full blocks (64/16 = 4)
    }

    #[test]
    fn test_find_longest_cache_hit_sliding_window() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = SlidingWindowManager::new(block_pool.clone(), 0);

        let hashes = vec![[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let spec = SlidingWindowSpec {
            block_size: 16,
            sliding_window: 48, // 3 blocks worth
        };

        // Test empty cache
        let result = manager.find_longest_cache_hit(
            &hashes,
            80,
            &[0],
            block_pool.clone(),
            Box::new(spec.clone()),
            false,
            1,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 0);

        // Test without cached blocks (since we can't access private fields)
        let result = manager.find_longest_cache_hit(
            &hashes,
            80,
            &[0],
            block_pool.clone(),
            Box::new(spec.clone()),
            false,
            1,
        );

        // Should not find any blocks since cache is empty
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 0);
    }

    #[test]
    fn test_find_longest_cache_hit_with_eagle() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = SlidingWindowManager::new(block_pool.clone(), 0);

        let hashes = vec![[0u8; 32], [1u8; 32], [2u8; 32]];
        let spec = SlidingWindowSpec {
            block_size: 16,
            sliding_window: 64,
        };

        // Test with eagle enabled (but no cached blocks)
        let result = manager.find_longest_cache_hit(
            &hashes,
            48,
            &[0],
            block_pool.clone(),
            Box::new(spec),
            true, // use_eagle
            1,
        );

        assert_eq!(result.len(), 1);
        // No cached blocks, so should be empty
        assert_eq!(result[0].len(), 0);
    }

    #[test]
    #[should_panic(expected = "DCP not supported for sliding window attention")]
    fn test_sliding_window_dcp_assertion() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = SlidingWindowManager::new(block_pool.clone(), 0);

        let spec = SlidingWindowSpec {
            block_size: 16,
            sliding_window: 128,
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
    fn test_sliding_window_null_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = SlidingWindowManager::new(block_pool.clone(), 0);

        // Allocate blocks
        manager.allocate_new_blocks("req", 256);

        // Manually set some blocks to null to test the break condition in remove_skipped_blocks
        let blocks = manager.base.req_to_blocks.get_mut("req").unwrap();
        let null_block = block_pool.borrow().get_null_block();
        blocks[2] = null_block.clone(); // Set middle block to null

        // Remove skipped blocks should stop at the null block
        manager.remove_skipped_blocks("req", 300);

        // Blocks before the null block should not be removed
        let final_blocks = manager.get_blocks("req").unwrap();
        assert!(final_blocks[0].borrow().block_id != null_block.borrow().block_id);
        assert!(final_blocks[1].borrow().block_id != null_block.borrow().block_id);
    }

    #[test]
    fn test_get_num_blocks_to_allocate() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let manager = SlidingWindowManager::new(block_pool.clone(), 0);

        // Test basic allocation calculation
        let num = manager.get_num_blocks_to_allocate("req", 48, &[]);
        assert_eq!(num, 3); // 48 / 16 = 3

        // Test with computed blocks
        let computed = vec![
            block_pool.borrow().get_null_block(),
            block_pool.borrow().get_null_block(),
        ];
        let num = manager.get_num_blocks_to_allocate("req", 48, &computed);
        assert_eq!(num, 1); // 3 needed - 2 computed = 1
    }
}
