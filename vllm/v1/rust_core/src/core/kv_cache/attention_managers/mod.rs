use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::block_pool::BlockPool;
use super::free_block_queue::{BlockHash, BlockRef};

pub mod chunked_local;
pub mod cross_attention;
pub mod full_attention;
pub mod mamba;
pub mod sliding_window;

pub use chunked_local::{ChunkedLocalAttentionManager, ChunkedLocalAttentionSpec};
pub use cross_attention::CrossAttentionManager;
pub use full_attention::{FullAttentionManager, FullAttentionSpec};
pub use mamba::{MambaManager, MambaSpec};
pub use sliding_window::{SlidingWindowManager, SlidingWindowSpec};

/// Trait for different KV cache specifications
pub trait KVCacheSpec: Any {
    fn block_size(&self) -> usize;
    fn spec_type(&self) -> AttentionType;
    fn as_any(&self) -> &dyn Any;
    fn box_clone(&self) -> Box<dyn KVCacheSpec>;
}

impl Clone for Box<dyn KVCacheSpec> {
    fn clone(&self) -> Box<dyn KVCacheSpec> {
        self.box_clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionType {
    FullAttention,
    SlidingWindow,
    ChunkedLocal,
    CrossAttention,
    Mamba,
}

/// Abstract base trait for single type KV cache managers
pub trait SingleTypeKVCacheManager {
    fn is_cross_attention(&self) -> bool {
        false
    }

    fn get_blocks(&self, request_id: &str) -> Option<Vec<BlockRef>>;

    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockRef],
    ) -> usize;

    fn save_new_computed_blocks(&mut self, request_id: &str, new_computed_blocks: &[BlockRef]);

    fn allocate_new_blocks(&mut self, request_id: &str, num_tokens: usize) -> Vec<BlockRef>;

    fn cache_blocks(
        &mut self,
        request_id: &str,
        request_block_hashes: &[BlockHash],
        num_tokens: usize,
    );

    fn free(&mut self, request_id: &str);

    fn get_num_common_prefix_blocks(&self, request_id: &str, num_running_requests: usize) -> usize;

    fn find_longest_cache_hit(
        &mut self,
        block_hashes: &[BlockHash],
        max_length: usize,
        kv_cache_group_ids: &[u32],
        block_pool: Rc<RefCell<BlockPool>>,
        kv_cache_spec: Box<dyn KVCacheSpec>,
        use_eagle: bool,
        dcp_world_size: usize,
    ) -> Vec<Vec<BlockRef>>;

    fn remove_skipped_blocks(&mut self, request_id: &str, num_computed_tokens: usize);
}

/// Base implementation struct that concrete managers will build upon
pub struct BaseKVCacheManager {
    pub block_size: usize,
    pub dcp_world_size: usize,
    pub block_pool: Rc<RefCell<BlockPool>>,
    pub req_to_blocks: HashMap<String, Vec<BlockRef>>,
    pub num_cached_blocks: HashMap<String, usize>,
    pub kv_cache_group_id: u32,
    pub null_block: BlockRef,
}

impl BaseKVCacheManager {
    pub fn new(
        kv_cache_spec: &dyn KVCacheSpec,
        block_pool: Rc<RefCell<BlockPool>>,
        kv_cache_group_id: u32,
        dcp_world_size: usize,
    ) -> Self {
        let mut block_size = kv_cache_spec.block_size();
        if dcp_world_size > 1 {
            block_size *= dcp_world_size;
        }

        let null_block = block_pool.borrow().get_null_block();

        Self {
            block_size,
            dcp_world_size,
            block_pool,
            req_to_blocks: HashMap::new(),
            num_cached_blocks: HashMap::new(),
            kv_cache_group_id,
            null_block,
        }
    }

    pub fn get_num_blocks_to_allocate_impl(
        &self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockRef],
    ) -> usize {
        let num_required_blocks = cdiv(num_tokens, self.block_size);
        let current_blocks = self.req_to_blocks.get(request_id).map_or(0, |b| b.len());
        let num_new_blocks = num_required_blocks
            .saturating_sub(new_computed_blocks.len())
            .saturating_sub(current_blocks);

        // Count evictable computed blocks
        let num_evictable_computed_blocks = new_computed_blocks
            .iter()
            .filter(|blk| {
                let b = blk.borrow();
                b.ref_cnt == 0 && !b.is_null
            })
            .count();

        num_new_blocks + num_evictable_computed_blocks
    }

    pub fn save_new_computed_blocks_impl(
        &mut self,
        request_id: &str,
        new_computed_blocks: &[BlockRef],
    ) {
        if !self.num_cached_blocks.contains_key(request_id) {
            // New request
            let req_blocks = self
                .req_to_blocks
                .entry(request_id.to_string())
                .or_insert_with(Vec::new);
            assert!(req_blocks.is_empty());
            req_blocks.extend(new_computed_blocks.iter().cloned());
            self.num_cached_blocks
                .insert(request_id.to_string(), new_computed_blocks.len());
        } else {
            // Running request - should not have new computed blocks
            assert!(new_computed_blocks.is_empty());
        }
    }

    pub fn allocate_new_blocks_impl(
        &mut self,
        request_id: &str,
        num_tokens: usize,
    ) -> Vec<BlockRef> {
        let req_blocks = self
            .req_to_blocks
            .entry(request_id.to_string())
            .or_insert_with(Vec::new);
        let num_required_blocks = cdiv(num_tokens, self.block_size);
        let num_new_blocks = num_required_blocks.saturating_sub(req_blocks.len());

        if num_new_blocks == 0 {
            Vec::new()
        } else {
            let new_blocks = self.block_pool.borrow_mut().get_new_blocks(num_new_blocks);
            req_blocks.extend(new_blocks.clone());
            new_blocks
        }
    }

    pub fn cache_blocks_impl(
        &mut self,
        request_id: &str,
        request_block_hashes: &[BlockHash],
        num_tokens: usize,
    ) {
        let num_cached_blocks = *self.num_cached_blocks.get(request_id).unwrap_or(&0);
        let num_full_blocks = num_tokens / self.block_size;

        if let Some(blocks) = self.req_to_blocks.get(request_id) {
            self.block_pool.borrow_mut().cache_full_blocks(
                request_block_hashes,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                self.block_size,
                self.kv_cache_group_id,
            );
        }

        self.num_cached_blocks
            .insert(request_id.to_string(), num_full_blocks);
    }

    pub fn free_impl(&mut self, request_id: &str) {
        if let Some(req_blocks) = self.req_to_blocks.remove(request_id) {
            // Free blocks in reverse order (tail blocks first)
            let ordered_blocks: Vec<BlockRef> = req_blocks.into_iter().rev().collect();
            self.block_pool.borrow_mut().free_blocks(ordered_blocks);
        }
        self.num_cached_blocks.remove(request_id);
    }
}

/// Factory function to create appropriate manager based on spec type
pub fn get_manager_for_kv_cache_spec(
    kv_cache_spec: &dyn KVCacheSpec,
    block_pool: Rc<RefCell<BlockPool>>,
    kv_cache_group_id: u32,
    dcp_world_size: usize,
) -> Box<dyn SingleTypeKVCacheManager> {
    match kv_cache_spec.spec_type() {
        AttentionType::FullAttention => Box::new(FullAttentionManager::new_with_spec(
            kv_cache_spec,
            block_pool,
            kv_cache_group_id,
            dcp_world_size,
        )),
        AttentionType::SlidingWindow => {
            if let Some(spec) = kv_cache_spec.as_any().downcast_ref::<SlidingWindowSpec>() {
                Box::new(SlidingWindowManager::new_with_spec(
                    spec,
                    block_pool,
                    kv_cache_group_id,
                    dcp_world_size,
                ))
            } else {
                panic!("Invalid spec type for SlidingWindow");
            }
        }
        AttentionType::CrossAttention => {
            Box::new(CrossAttentionManager::new(block_pool, kv_cache_group_id))
        }
        AttentionType::ChunkedLocal => {
            if let Some(spec) = kv_cache_spec
                .as_any()
                .downcast_ref::<ChunkedLocalAttentionSpec>()
            {
                Box::new(ChunkedLocalAttentionManager::new_with_spec(
                    spec,
                    block_pool,
                    kv_cache_group_id,
                    dcp_world_size,
                ))
            } else {
                panic!("Invalid spec type for ChunkedLocal");
            }
        }
        AttentionType::Mamba => {
            if let Some(spec) = kv_cache_spec.as_any().downcast_ref::<MambaSpec>() {
                Box::new(MambaManager::new_with_spec(
                    spec,
                    block_pool,
                    kv_cache_group_id,
                    dcp_world_size,
                ))
            } else {
                panic!("Invalid spec type for Mamba");
            }
        }
    }
}

/// Utility function for ceiling division
#[inline]
pub fn cdiv(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdiv_edge_cases() {
        assert_eq!(cdiv(10, 3), 4);
        assert_eq!(cdiv(9, 3), 3);
        assert_eq!(cdiv(0, 5), 0);
        assert_eq!(cdiv(1, 1), 1);
        assert_eq!(cdiv(7, 10), 1);
    }

    #[test]
    fn test_base_manager_allocation_edge_cases() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let mut base = BaseKVCacheManager::new(&spec, block_pool.clone(), 0, 1);

        // Test allocating 0 tokens
        let blocks = base.allocate_new_blocks_impl("req1", 0);
        assert_eq!(blocks.len(), 0);

        // Test incremental allocation
        let blocks1 = base.allocate_new_blocks_impl("req2", 20); // Needs 2 blocks
        assert_eq!(blocks1.len(), 2);
        let blocks2 = base.allocate_new_blocks_impl("req2", 40); // Needs 3 total, already has 2
        assert_eq!(blocks2.len(), 1);

        // Test allocation with partial blocks
        let blocks3 = base.allocate_new_blocks_impl("req3", 15); // Less than one block
        assert_eq!(blocks3.len(), 1);
    }

    #[test]
    fn test_base_manager_get_num_blocks_with_evictable() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let base = BaseKVCacheManager::new(&spec, block_pool.clone(), 0, 1);

        // Create some mock computed blocks using actual blocks
        let blocks = block_pool.borrow_mut().get_new_blocks(2);
        let evictable_block = blocks[0].clone();
        evictable_block.borrow_mut().ref_cnt = 0;
        evictable_block.borrow_mut().is_null = false;

        let non_evictable_block = blocks[1].clone();
        non_evictable_block.borrow_mut().ref_cnt = 1;
        non_evictable_block.borrow_mut().is_null = false;

        let computed_blocks = vec![evictable_block, non_evictable_block];

        // Should count evictable blocks
        let num = base.get_num_blocks_to_allocate_impl("new_req", 32, &computed_blocks);
        assert_eq!(num, 1); // 2 needed - 2 computed + 1 evictable = 1
    }

    #[test]
    fn test_base_manager_save_computed_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let mut base = BaseKVCacheManager::new(&spec, block_pool.clone(), 0, 1);

        let block1 = block_pool.borrow().get_null_block();
        let block2 = block_pool.borrow().get_null_block();
        let blocks = vec![block1, block2];

        // Save to new request
        base.save_new_computed_blocks_impl("new_req", &blocks);
        assert_eq!(base.req_to_blocks.get("new_req").unwrap().len(), 2);
        assert_eq!(*base.num_cached_blocks.get("new_req").unwrap(), 2);

        // Save empty blocks to existing request (should assert empty)
        base.save_new_computed_blocks_impl("new_req", &[]);
        assert_eq!(base.req_to_blocks.get("new_req").unwrap().len(), 2);
    }

    #[test]
    #[should_panic]
    fn test_base_manager_save_computed_blocks_panic() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let mut base = BaseKVCacheManager::new(&spec, block_pool.clone(), 0, 1);

        let block = block_pool.borrow().get_null_block();
        base.save_new_computed_blocks_impl("req", &[block.clone()]);

        // Should panic when trying to save blocks to existing request
        base.save_new_computed_blocks_impl("req", &[block]);
    }

    #[test]
    fn test_base_manager_cache_blocks() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, true, false))); // Enable caching
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let mut base = BaseKVCacheManager::new(&spec, block_pool.clone(), 0, 1);

        // First allocate some blocks
        base.allocate_new_blocks_impl("req", 32);

        // Cache with partial blocks
        let hashes = vec![[0u8; 32], [1u8; 32]];
        base.cache_blocks_impl("req", &hashes, 20); // 20 tokens = 1 full block
        assert_eq!(*base.num_cached_blocks.get("req").unwrap(), 1);

        // Cache with no blocks allocated (should handle gracefully)
        base.cache_blocks_impl("nonexistent", &hashes, 32);
        assert_eq!(*base.num_cached_blocks.get("nonexistent").unwrap(), 2);
    }

    #[test]
    fn test_base_manager_free_edge_cases() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let mut base = BaseKVCacheManager::new(&spec, block_pool, 0, 1);

        // Free non-existent request (should not panic)
        base.free_impl("nonexistent");
        assert!(!base.req_to_blocks.contains_key("nonexistent"));

        // Allocate and free normally
        base.allocate_new_blocks_impl("req", 32);
        assert!(base.req_to_blocks.contains_key("req"));
        base.free_impl("req");
        assert!(!base.req_to_blocks.contains_key("req"));

        // Double free (should not panic)
        base.free_impl("req");
    }

    #[test]
    fn test_base_manager_with_dcp() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));
        let spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let base = BaseKVCacheManager::new(&spec, block_pool, 0, 4); // DCP world size = 4

        // Block size should be multiplied by DCP world size
        assert_eq!(base.block_size, 64);
        assert_eq!(base.dcp_world_size, 4);
    }

    #[test]
    fn test_factory_function_all_types() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));

        // Test FullAttention
        let full_spec = FullAttentionSpec {
            block_size: 16,
            sliding_window: None,
        };
        let manager = get_manager_for_kv_cache_spec(&full_spec, block_pool.clone(), 0, 1);
        assert!(!manager.is_cross_attention());

        // Test SlidingWindow
        let sliding_spec = SlidingWindowSpec {
            block_size: 16,
            sliding_window: 128,
        };
        let manager = get_manager_for_kv_cache_spec(&sliding_spec, block_pool.clone(), 0, 1);
        assert!(!manager.is_cross_attention());

        // Test CrossAttention
        struct CrossAttentionSpec;
        impl KVCacheSpec for CrossAttentionSpec {
            fn block_size(&self) -> usize {
                16
            }
            fn spec_type(&self) -> AttentionType {
                AttentionType::CrossAttention
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn box_clone(&self) -> Box<dyn KVCacheSpec> {
                Box::new(CrossAttentionSpec)
            }
        }
        let cross_spec = CrossAttentionSpec;
        let manager = get_manager_for_kv_cache_spec(&cross_spec, block_pool.clone(), 0, 1);
        assert!(manager.is_cross_attention());
    }

    #[test]
    #[should_panic(expected = "Invalid spec type for Mamba")]
    fn test_factory_unsupported_type() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));

        // Create a spec that returns Mamba type but isn't actually MambaSpec
        struct InvalidMambaSpec;
        impl KVCacheSpec for InvalidMambaSpec {
            fn block_size(&self) -> usize {
                16
            }
            fn spec_type(&self) -> AttentionType {
                AttentionType::Mamba
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn box_clone(&self) -> Box<dyn KVCacheSpec> {
                Box::new(InvalidMambaSpec)
            }
        }

        let spec = InvalidMambaSpec;
        get_manager_for_kv_cache_spec(&spec, block_pool, 0, 1);
    }

    #[test]
    #[should_panic(expected = "Invalid spec type for SlidingWindow")]
    fn test_factory_wrong_downcast() {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(100, false, false)));

        // Create a spec that reports SlidingWindow type but isn't actually SlidingWindowSpec
        struct FakeSliding;
        impl KVCacheSpec for FakeSliding {
            fn block_size(&self) -> usize {
                16
            }
            fn spec_type(&self) -> AttentionType {
                AttentionType::SlidingWindow
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn box_clone(&self) -> Box<dyn KVCacheSpec> {
                Box::new(FakeSliding)
            }
        }

        let spec = FakeSliding;
        get_manager_for_kv_cache_spec(&spec, block_pool, 0, 1);
    }
}
