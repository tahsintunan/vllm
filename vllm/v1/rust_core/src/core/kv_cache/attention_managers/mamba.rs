use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::{cdiv, AttentionType, BaseKVCacheManager, KVCacheSpec, SingleTypeKVCacheManager};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};

#[derive(Debug, Clone)]
pub struct MambaSpec {
    pub block_size: usize,
    pub num_speculative_blocks: usize,
    pub mamba_type: String,
}

impl KVCacheSpec for MambaSpec {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn spec_type(&self) -> AttentionType {
        AttentionType::Mamba
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn box_clone(&self) -> Box<dyn KVCacheSpec> {
        Box::new(self.clone())
    }
}

pub struct MambaManager {
    base: BaseKVCacheManager,
    num_speculative_blocks: usize,
}

impl MambaManager {
    pub fn new(block_pool: Rc<RefCell<BlockPool>>, kv_cache_group_id: u32) -> Self {
        let spec = MambaSpec {
            block_size: 16,
            num_speculative_blocks: 0,
            mamba_type: "mamba2".to_string(),
        };
        Self {
            base: BaseKVCacheManager::new(&spec, block_pool, kv_cache_group_id, 1),
            num_speculative_blocks: spec.num_speculative_blocks,
        }
    }

    pub fn new_with_spec(
        kv_cache_spec: &MambaSpec,
        block_pool: Rc<RefCell<BlockPool>>,
        kv_cache_group_id: u32,
        dcp_world_size: usize,
    ) -> Self {
        assert_eq!(dcp_world_size, 1, "DCP not support mamba now.");
        Self {
            base: BaseKVCacheManager::new(
                kv_cache_spec,
                block_pool,
                kv_cache_group_id,
                dcp_world_size,
            ),
            num_speculative_blocks: kv_cache_spec.num_speculative_blocks,
        }
    }
}

impl SingleTypeKVCacheManager for MambaManager {
    fn get_blocks(&self, request_id: &str) -> Option<Vec<BlockRef>> {
        self.base.req_to_blocks.get(request_id).map(|b| b.clone())
    }

    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockRef],
    ) -> usize {
        // Adjust num_tokens for speculative blocks
        let adjusted_tokens = if self.num_speculative_blocks > 0 {
            num_tokens + (self.base.block_size * self.num_speculative_blocks)
        } else {
            num_tokens
        };

        let num_required_blocks = cdiv(adjusted_tokens, self.base.block_size);
        let existing_blocks = self
            .base
            .req_to_blocks
            .get(request_id)
            .map(|blocks| blocks.len())
            .unwrap_or(0);
        let num_new_blocks = num_required_blocks - new_computed_blocks.len() - existing_blocks;

        // If a computed block of a request is an eviction candidate (in the
        // free queue and ref_cnt == 0), it will be changed from a free block
        // to a computed block when the request is allocated, so we also count
        // it as needed to be allocated.
        let num_evictable_computed_blocks = new_computed_blocks
            .iter()
            .filter(|blk| blk.borrow().ref_cnt == 0 && blk.borrow().block_id != self.base.null_block.borrow().block_id)
            .count();

        num_new_blocks + num_evictable_computed_blocks
    }

    fn save_new_computed_blocks(&mut self, request_id: &str, new_computed_blocks: &[BlockRef]) {
        self.base
            .save_new_computed_blocks_impl(request_id, new_computed_blocks)
    }

    fn allocate_new_blocks(&mut self, request_id: &str, num_tokens: usize) -> Vec<BlockRef> {
        // Allocate extra `num_speculative_blocks` blocks for
        // speculative decoding (MTP/EAGLE) with linear attention.
        let adjusted_tokens = if self.num_speculative_blocks > 0 {
            num_tokens + (self.base.block_size * self.num_speculative_blocks)
        } else {
            num_tokens
        };
        self.base.allocate_new_blocks_impl(request_id, adjusted_tokens)
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
        // Mamba doesn't support prefix sharing
        0
    }

    fn find_longest_cache_hit(
        &mut self,
        _block_hashes: &[BlockHash],
        _max_length: usize,
        kv_cache_group_ids: &[u32],
        _block_pool: Rc<RefCell<BlockPool>>,
        kv_cache_spec: Box<dyn KVCacheSpec>,
        _use_eagle: bool,
        dcp_world_size: usize,
    ) -> Vec<Vec<BlockRef>> {
        assert!(
            kv_cache_spec.as_any().downcast_ref::<MambaSpec>().is_some(),
            "MambaManager can only be used for mamba groups"
        );
        assert_eq!(dcp_world_size, 1, "DCP not support mamba now.");

        // Prefix caching is not supported for mamba now. Always return empty list.
        vec![Vec::new(); kv_cache_group_ids.len()]
    }

    fn remove_skipped_blocks(&mut self, _request_id: &str, _num_computed_tokens: usize) {
        // Each request will always have 1 block at this moment, so no need to
        // remove blocks.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_spec() {
        let spec = MambaSpec {
            block_size: 32,
            num_speculative_blocks: 2,
            mamba_type: "mamba2".to_string(),
        };

        assert_eq!(spec.block_size(), 32);
        assert_eq!(spec.spec_type(), AttentionType::Mamba);
        assert_eq!(spec.num_speculative_blocks, 2);
        assert_eq!(spec.mamba_type, "mamba2");

        // Test cloning
        let cloned_spec = spec.box_clone();
        assert_eq!(cloned_spec.block_size(), 32);
    }

    #[test]
    fn test_mamba_manager_basic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = MambaManager::new(block_pool, 0);

        // Test allocation
        let blocks = manager.allocate_new_blocks("test_req", 32);
        assert_eq!(blocks.len(), 2); // 32 tokens / 16 block_size = 2 blocks

        // Test getting blocks
        let retrieved_blocks = manager.get_blocks("test_req");
        assert!(retrieved_blocks.is_some());
        assert_eq!(retrieved_blocks.unwrap().len(), 2);

        // Test that prefix blocks returns 0 for mamba
        let prefix_blocks = manager.get_num_common_prefix_blocks("test_req", 2);
        assert_eq!(prefix_blocks, 0);

        // Test freeing
        manager.free("test_req");
        assert!(manager.get_blocks("test_req").is_none());
    }

    #[test]
    fn test_mamba_with_speculative_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let spec = MambaSpec {
            block_size: 16,
            num_speculative_blocks: 2, // 2 extra blocks
            mamba_type: "mamba2".to_string(),
        };
        let mut manager = MambaManager::new_with_spec(&spec, block_pool, 0, 1);

        // Allocate blocks for 16 tokens
        // Should allocate 16 + (16 * 2) = 48 tokens worth = 3 blocks
        let blocks = manager.allocate_new_blocks("req", 16);
        assert_eq!(blocks.len(), 3); // (16 + 32) / 16 = 3 blocks

        // Test get_num_blocks_to_allocate with speculative blocks
        let num = manager.get_num_blocks_to_allocate("new_req", 16, &[]);
        assert_eq!(num, 3); // (16 + 32) / 16 = 3 blocks
    }

    #[test]
    fn test_mamba_find_longest_cache_hit() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = MambaManager::new(block_pool.clone(), 0);

        let spec = MambaSpec {
            block_size: 16,
            num_speculative_blocks: 0,
            mamba_type: "mamba2".to_string(),
        };

        // Should always return empty for mamba
        let result = manager.find_longest_cache_hit(
            &[[0u8; 32], [1u8; 32]],
            32,
            &[0, 1],
            block_pool,
            Box::new(spec),
            false,
            1,
        );

        assert_eq!(result.len(), 2);
        assert!(result[0].is_empty());
        assert!(result[1].is_empty());
    }

    #[test]
    #[should_panic(expected = "DCP not support mamba now")]
    fn test_mamba_dcp_panic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = MambaManager::new(block_pool.clone(), 0);

        let spec = MambaSpec {
            block_size: 16,
            num_speculative_blocks: 0,
            mamba_type: "mamba2".to_string(),
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
    fn test_mamba_remove_skipped_blocks_noop() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = MambaManager::new(block_pool, 0);

        // Allocate blocks
        manager.allocate_new_blocks("req", 64);
        let blocks_before = manager.get_blocks("req").unwrap().len();

        // Remove skipped blocks should do nothing for mamba
        manager.remove_skipped_blocks("req", 100);

        let blocks_after = manager.get_blocks("req").unwrap().len();
        assert_eq!(blocks_before, blocks_after);
    }

    #[test]
    fn test_mamba_cache_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false)));
        let mut manager = MambaManager::new(block_pool, 0);

        manager.allocate_new_blocks("req", 32);
        let hashes = vec![[0u8; 32], [1u8; 32]];

        // Cache blocks
        manager.cache_blocks("req", &hashes, 32);

        // Should work without issues
        assert_eq!(manager.get_blocks("req").unwrap().len(), 2);
    }

    #[test]
    fn test_mamba_save_new_computed_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let mut manager = MambaManager::new(block_pool.clone(), 0);

        // Create computed blocks
        let blocks: Vec<_> = (0..2)
            .map(|_| block_pool.borrow_mut().get_new_blocks(1)[0].clone())
            .collect();

        // Save for new request
        manager.save_new_computed_blocks("new_req", &blocks);
        assert_eq!(manager.get_blocks("new_req").unwrap().len(), 2);
    }

    #[test]
    fn test_mamba_get_num_blocks_to_allocate_with_evictable() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let manager = MambaManager::new(block_pool.clone(), 0);

        // Create blocks with ref_cnt = 0 (evictable)
        let mut evictable_blocks = Vec::new();
        for _ in 0..2 {
            let block = block_pool.borrow_mut().get_new_blocks(1)[0].clone();
            block.borrow_mut().ref_cnt = 0;
            evictable_blocks.push(block);
        }

        // Test with evictable blocks
        let num = manager.get_num_blocks_to_allocate("req", 48, &evictable_blocks);
        // Need 3 blocks total (48/16), have 2 evictable, so need 3 - 2 + 2 = 3
        assert_eq!(num, 3);
    }
}