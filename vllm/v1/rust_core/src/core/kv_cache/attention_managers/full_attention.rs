use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::{AttentionType, BaseKVCacheManager, KVCacheSpec, SingleTypeKVCacheManager};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};

#[derive(Debug, Clone)]
pub struct FullAttentionSpec {
    pub block_size: usize,
    pub sliding_window: Option<usize>,
}

impl KVCacheSpec for FullAttentionSpec {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn spec_type(&self) -> AttentionType {
        AttentionType::FullAttention
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn box_clone(&self) -> Box<dyn KVCacheSpec> {
        Box::new(self.clone())
    }
}

pub struct FullAttentionManager {
    base: BaseKVCacheManager,
}

impl FullAttentionManager {
    pub fn new(block_pool: Rc<RefCell<BlockPool>>, kv_cache_group_id: u32) -> Self {
        // Use a default spec for simple construction
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        Self {
            base: BaseKVCacheManager::new(&spec, block_pool, kv_cache_group_id, 1),
        }
    }

    pub fn new_with_spec(
        kv_cache_spec: &dyn KVCacheSpec,
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
        }
    }
}

impl SingleTypeKVCacheManager for FullAttentionManager {
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

    fn get_num_common_prefix_blocks(&self, request_id: &str, num_running_requests: usize) -> usize {
        if let Some(blocks) = self.base.req_to_blocks.get(request_id) {
            let mut num_common_blocks = 0;
            for block in blocks {
                if block.borrow().ref_cnt == num_running_requests as u32 {
                    num_common_blocks += 1;
                } else {
                    break;
                }
            }
            num_common_blocks
        } else {
            0
        }
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
        let mut block_size = kv_cache_spec.block_size();
        if dcp_world_size > 1 {
            block_size *= dcp_world_size;
        }

        let max_num_blocks = max_length / block_size;
        let mut computed_blocks: Vec<Vec<BlockRef>> = vec![Vec::new(); kv_cache_group_ids.len()];

        for block_hash in block_hashes.iter().take(max_num_blocks) {
            if let Some(cached_blocks) = block_pool
                .borrow()
                .get_cached_block(block_hash, kv_cache_group_ids)
            {
                for (computed, cached) in computed_blocks.iter_mut().zip(cached_blocks.iter()) {
                    computed.push(cached.clone());
                }
            } else {
                break;
            }
        }

        // Drop last block if eagle is enabled
        if use_eagle && !computed_blocks[0].is_empty() {
            for computed in &mut computed_blocks {
                computed.pop();
            }
        }

        computed_blocks
    }

    fn remove_skipped_blocks(&mut self, _request_id: &str, _num_computed_tokens: usize) {
        // No need to remove blocks for full attention
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_attention_manager_basic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = FullAttentionManager::new(block_pool, 0);

        // Test allocation
        let blocks = manager.allocate_new_blocks("test_req", 32);
        assert_eq!(blocks.len(), 2); // 32 tokens / 16 block_size = 2 blocks

        // Test getting blocks
        let retrieved_blocks = manager.get_blocks("test_req");
        assert!(retrieved_blocks.is_some());
        assert_eq!(retrieved_blocks.unwrap().len(), 2);

        // Test freeing
        manager.free("test_req");
        assert!(manager.get_blocks("test_req").is_none());
    }

    #[test]
    fn test_full_attention_spec() {
        let spec = FullAttentionSpec {
            block_size: 32,
            sliding_window: Some(256),
        };

        assert_eq!(spec.block_size(), 32);
        assert_eq!(spec.spec_type(), AttentionType::FullAttention);

        // Test cloning
        let cloned_spec = spec.box_clone();
        assert_eq!(cloned_spec.block_size(), 32);
    }

    #[test]
    fn test_get_num_common_prefix_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = FullAttentionManager::new(block_pool.clone(), 0);

        // Allocate blocks
        let blocks = manager.allocate_new_blocks("req1", 64);

        // Set ref counts to simulate shared blocks
        blocks[0].borrow_mut().ref_cnt = 3;
        blocks[1].borrow_mut().ref_cnt = 3;
        blocks[2].borrow_mut().ref_cnt = 1; // Not shared
        blocks[3].borrow_mut().ref_cnt = 1;

        let num_common = manager.get_num_common_prefix_blocks("req1", 3);
        assert_eq!(num_common, 2); // First 2 blocks have ref_cnt == 3

        // Test non-existent request
        let num_common = manager.get_num_common_prefix_blocks("nonexistent", 2);
        assert_eq!(num_common, 0);
    }

    #[test]
    fn test_save_new_computed_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = FullAttentionManager::new(block_pool.clone(), 0);

        // Create computed blocks
        let mut blocks = Vec::new();
        for _ in 0..3 {
            blocks.push(block_pool.borrow_mut().get_new_blocks(1)[0].clone());
        }

        // Save computed blocks for new request
        manager.save_new_computed_blocks("new_req", &blocks);
        let retrieved = manager.get_blocks("new_req").unwrap();
        assert_eq!(retrieved.len(), 3);

        // Save empty blocks for existing request (should be fine)
        manager.save_new_computed_blocks("new_req", &[]);
        assert_eq!(manager.get_blocks("new_req").unwrap().len(), 3);
    }

    #[test]
    fn test_get_num_blocks_to_allocate() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let manager = FullAttentionManager::new(block_pool.clone(), 0);

        // Test with no existing blocks
        let num = manager.get_num_blocks_to_allocate("new_req", 48, &[]);
        assert_eq!(num, 3); // 48 tokens / 16 block_size = 3 blocks

        // Test with computed blocks
        let computed_blocks = vec![
            block_pool.borrow().get_null_block(),
            block_pool.borrow().get_null_block(),
        ];
        let num = manager.get_num_blocks_to_allocate("new_req", 48, &computed_blocks);
        assert_eq!(num, 1); // 3 needed - 2 computed = 1
    }

    #[test]
    fn test_cache_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false))); // Enable caching
        let mut manager = FullAttentionManager::new(block_pool, 0);

        // Allocate blocks
        manager.allocate_new_blocks("req", 48);

        // Create block hashes (need enough for all blocks that could be cached)
        let hashes = vec![[0u8; 32], [1u8; 32], [2u8; 32]];

        // Cache partial blocks first
        manager.cache_blocks("req", &hashes, 32); // 2 full blocks (32/16 = 2)

        // Then cache all blocks
        manager.cache_blocks("req", &hashes, 48); // 3 full blocks (48/16 = 3)
    }

    #[test]
    fn test_find_longest_cache_hit() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = FullAttentionManager::new(block_pool.clone(), 0);

        // Create some block hashes
        let hashes = vec![[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32]];

        // Test with no cached blocks (cache miss)
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let result = manager.find_longest_cache_hit(
            &hashes,
            64,   // max_length
            &[0], // kv_cache_group_ids
            block_pool.clone(),
            Box::new(spec.clone()),
            false, // use_eagle
            1,     // dcp_world_size
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 0); // No cache hits

        // Test with use_eagle enabled (should drop last block)
        // Since we can't manually manipulate the cache (private field),
        // we'll just test the basic behavior without cached blocks
        let mut manager2 = FullAttentionManager::new(block_pool.clone(), 0);

        let result = manager2.find_longest_cache_hit(
            &hashes,
            64,
            &[0],
            block_pool.clone(),
            Box::new(spec.clone()),
            true, // use_eagle = true
            1,
        );
        assert_eq!(result.len(), 1);
        // No cached blocks, so should be empty
        assert_eq!(result[0].len(), 0);
    }

    #[test]
    fn test_find_longest_cache_hit_with_dcp() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = FullAttentionManager::new_with_spec(
            &FullAttentionSpec {
                block_size: 16,
                sliding_window: None,
            },
            block_pool.clone(),
            0,
            2, // DCP world size = 2
        );

        let hashes = vec![[0u8; 32], [1u8; 32]];
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };

        // With DCP > 1, block size should be adjusted
        let result = manager.find_longest_cache_hit(
            &hashes,
            64,
            &[0],
            block_pool.clone(),
            Box::new(spec),
            false,
            2, // DCP world size
        );
        assert_eq!(result.len(), 1);
        // Block size becomes 16 * 2 = 32, so 64/32 = 2 blocks max
    }

    #[test]
    fn test_multiple_kv_cache_groups() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = FullAttentionManager::new(block_pool.clone(), 0);

        let hashes = vec![[0u8; 32], [1u8; 32]];
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };

        // Test with multiple cache group IDs
        let result = manager.find_longest_cache_hit(
            &hashes,
            32,
            &[0, 1, 2], // Multiple groups
            block_pool.clone(),
            Box::new(spec),
            false,
            1,
        );
        assert_eq!(result.len(), 3); // One result per group
        for group_result in result {
            assert_eq!(group_result.len(), 0); // No cached blocks
        }
    }
}
