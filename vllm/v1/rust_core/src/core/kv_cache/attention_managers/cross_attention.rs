use std::cell::RefCell;
use std::rc::Rc;

use super::{
    BaseKVCacheManager, KVCacheSpec, SingleTypeKVCacheManager,
};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};
use crate::core::kv_cache::attention_managers::full_attention::FullAttentionSpec;

/// CrossAttentionManager stub implementation
pub struct CrossAttentionManager {
    base: BaseKVCacheManager,
}

impl CrossAttentionManager {
    pub fn new(block_pool: Rc<RefCell<BlockPool>>, kv_cache_group_id: u32) -> Self {
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        }; // Default spec
        Self {
            base: BaseKVCacheManager::new(&spec, block_pool, kv_cache_group_id, 1),
        }
    }
}

impl SingleTypeKVCacheManager for CrossAttentionManager {
    fn is_cross_attention(&self) -> bool {
        true
    }

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
        0 // Cross attention doesn't share prefixes
    }

    fn find_longest_cache_hit(
        &mut self,
        _block_hashes: &[BlockHash],
        _max_length: usize,
        kv_cache_group_ids: &[u32],
        _block_pool: Rc<RefCell<BlockPool>>,
        _kv_cache_spec: Box<dyn KVCacheSpec>,
        _use_eagle: bool,
        _dcp_world_size: usize,
    ) -> Vec<Vec<BlockRef>> {
        // Cross attention doesn't use caching
        vec![Vec::new(); kv_cache_group_ids.len()]
    }

    fn remove_skipped_blocks(&mut self, _request_id: &str, _num_computed_tokens: usize) {
        // No blocks to remove for cross attention
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_attention_manager_basic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = CrossAttentionManager::new(block_pool.clone(), 0);

        // Test that is_cross_attention returns true
        assert!(manager.is_cross_attention());

        // Test allocation
        let blocks = manager.allocate_new_blocks("test_req", 32);
        assert_eq!(blocks.len(), 2); // 32 tokens / 16 block_size = 2 blocks

        // Test getting blocks
        let retrieved_blocks = manager.get_blocks("test_req");
        assert!(retrieved_blocks.is_some());
        assert_eq!(retrieved_blocks.unwrap().len(), 2);

        // Test that prefix blocks returns 0 for cross attention
        let prefix_blocks = manager.get_num_common_prefix_blocks("test_req", 2);
        assert_eq!(prefix_blocks, 0);

        // Test that find_longest_cache_hit returns empty for cross attention
        let cache_hit = manager.find_longest_cache_hit(
            &[],
            100,
            &[0],
            block_pool.clone(),
            Box::new(FullAttentionSpec { block_size: 16, sliding_window: None }),
            false,
            1,
        );
        assert_eq!(cache_hit.len(), 1);
        assert!(cache_hit[0].is_empty());

        // Test freeing
        manager.free("test_req");
        assert!(manager.get_blocks("test_req").is_none());
    }

    #[test]
    fn test_save_new_computed_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = CrossAttentionManager::new(block_pool.clone(), 0);

        // Create computed blocks
        let blocks: Vec<_> = (0..2)
            .map(|_| block_pool.borrow_mut().get_new_blocks(1)[0].clone())
            .collect();

        // Save for new request
        manager.save_new_computed_blocks("new_req", &blocks);
        assert_eq!(manager.get_blocks("new_req").unwrap().len(), 2);

        // Save empty for existing request
        manager.save_new_computed_blocks("new_req", &[]);
        assert_eq!(manager.get_blocks("new_req").unwrap().len(), 2);
    }

    #[test]
    fn test_cache_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = CrossAttentionManager::new(block_pool, 0);

        // Allocate blocks
        manager.allocate_new_blocks("req", 48);
        let hashes = vec![[0u8; 32], [1u8; 32], [2u8; 32]];

        // Cache partial blocks
        manager.cache_blocks("req", &hashes, 32); // 2 full blocks (32/16 = 2)

        // Cache all blocks
        manager.cache_blocks("req", &hashes, 48); // 3 full blocks (48/16 = 3)
    }

    #[test]
    fn test_get_num_blocks_to_allocate() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let manager = CrossAttentionManager::new(block_pool.clone(), 0);

        // Test without existing blocks
        let num = manager.get_num_blocks_to_allocate("req", 32, &[]);
        assert_eq!(num, 2); // 32 / 16 = 2

        // Test with computed blocks
        let computed = vec![block_pool.borrow().get_null_block()];
        let num = manager.get_num_blocks_to_allocate("req", 32, &computed);
        assert_eq!(num, 1); // 2 needed - 1 computed = 1
    }

    #[test]
    fn test_find_longest_cache_hit_always_empty() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = CrossAttentionManager::new(block_pool.clone(), 0);

        // Even with cached blocks, cross attention should return empty
        let hashes = vec![[0u8; 32], [1u8; 32]];

        // Test with multiple groups
        let result = manager.find_longest_cache_hit(
            &hashes,
            32,
            &[0, 1, 2],
            block_pool.clone(),
            Box::new(FullAttentionSpec { block_size: 16, sliding_window: None }),
            true, // use_eagle shouldn't matter
            1,
        );

        assert_eq!(result.len(), 3);
        for group_result in result {
            assert!(group_result.is_empty());
        }
    }

    #[test]
    fn test_remove_skipped_blocks_noop() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = CrossAttentionManager::new(block_pool, 0);

        // Allocate blocks
        manager.allocate_new_blocks("req", 64);
        let blocks_before = manager.get_blocks("req").unwrap().len();

        // Remove skipped blocks should do nothing for cross attention
        manager.remove_skipped_blocks("req", 100);

        let blocks_after = manager.get_blocks("req").unwrap().len();
        assert_eq!(blocks_before, blocks_after);
    }

    #[test]
    fn test_incremental_allocation() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = CrossAttentionManager::new(block_pool, 0);

        // Initial allocation
        let blocks1 = manager.allocate_new_blocks("req", 20);
        assert_eq!(blocks1.len(), 2); // 20 / 16 = 2 blocks

        // Incremental allocation
        let blocks2 = manager.allocate_new_blocks("req", 50);
        assert_eq!(blocks2.len(), 2); // 50 / 16 = 4 total, already has 2

        // Total should be 4
        assert_eq!(manager.get_blocks("req").unwrap().len(), 4);
    }

    #[test]
    fn test_edge_cases() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = CrossAttentionManager::new(block_pool, 0);

        // Allocate 0 tokens
        let blocks = manager.allocate_new_blocks("empty_req", 0);
        assert_eq!(blocks.len(), 0);

        // Get blocks for non-existent request
        assert!(manager.get_blocks("nonexistent").is_none());

        // Free non-existent request (should not panic)
        manager.free("nonexistent");

        // Double free (should not panic)
        manager.allocate_new_blocks("req", 16);
        manager.free("req");
        manager.free("req");
    }
}