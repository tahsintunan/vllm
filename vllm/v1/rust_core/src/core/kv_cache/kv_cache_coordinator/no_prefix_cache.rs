use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::{BaseKVCacheCoordinator, KVCacheConfig, KVCacheCoordinator};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};
use crate::core::types::request::Request;

pub struct KVCacheCoordinatorNoPrefixCache {
    base: BaseKVCacheCoordinator,
    num_single_type_manager: usize,
}

impl KVCacheCoordinatorNoPrefixCache {
    pub fn new(
        kv_cache_config: KVCacheConfig,
        max_model_len: usize,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
    ) -> Self {
        let base = BaseKVCacheCoordinator::new(
            kv_cache_config,
            max_model_len,
            use_eagle,
            false, // enable_caching = false
            enable_kv_cache_events,
            dcp_world_size,
        );
        let num_single_type_manager = base.single_type_managers.len();

        Self {
            base,
            num_single_type_manager,
        }
    }
}

impl KVCacheCoordinator for KVCacheCoordinatorNoPrefixCache {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_block_pool(&self) -> Rc<RefCell<BlockPool>> {
        self.base.block_pool.clone()
    }
    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[Vec<BlockRef>],
        num_encoder_tokens: usize,
    ) -> usize {
        let mut num_blocks_to_allocate = 0;
        for (i, manager) in self.base.single_type_managers.iter().enumerate() {
            if manager.is_cross_attention() {
                num_blocks_to_allocate +=
                    manager.get_num_blocks_to_allocate(request_id, num_encoder_tokens, &[]);
            } else {
                let blocks = if i < new_computed_blocks.len() {
                    &new_computed_blocks[i]
                } else {
                    &Vec::new()
                };
                num_blocks_to_allocate +=
                    manager.get_num_blocks_to_allocate(request_id, num_tokens, blocks);
            }
        }
        num_blocks_to_allocate
    }

    fn save_new_computed_blocks(
        &mut self,
        request_id: &str,
        new_computed_blocks: &[Vec<BlockRef>],
    ) {
        for (i, manager) in self.base.single_type_managers.iter_mut().enumerate() {
            if i < new_computed_blocks.len() {
                manager.save_new_computed_blocks(request_id, &new_computed_blocks[i]);
            }
        }
    }

    fn allocate_new_blocks(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        num_encoder_tokens: usize,
    ) -> Vec<Vec<BlockRef>> {
        self.base
            .single_type_managers
            .iter_mut()
            .map(|manager| {
                if manager.is_cross_attention() {
                    manager.allocate_new_blocks(request_id, num_encoder_tokens)
                } else {
                    manager.allocate_new_blocks(request_id, num_tokens)
                }
            })
            .collect()
    }

    fn cache_blocks(&mut self, request: &Request, num_computed_tokens: usize) {
        for manager in self.base.single_type_managers.iter_mut() {
            manager.cache_blocks(
                &request.request_id,
                &request.block_hashes,
                num_computed_tokens,
            );
        }
    }

    fn free(&mut self, request_id: &str) {
        for manager in self.base.single_type_managers.iter_mut() {
            manager.free(request_id);
        }
    }

    fn get_num_common_prefix_blocks(
        &self,
        _request_id: &str,
        _num_running_requests: usize,
    ) -> Vec<usize> {
        vec![0; self.num_single_type_manager]
    }

    fn remove_skipped_blocks(&mut self, request_id: &str, num_computed_tokens: usize) {
        for manager in self.base.single_type_managers.iter_mut() {
            manager.remove_skipped_blocks(request_id, num_computed_tokens);
        }
    }

    fn get_blocks(&self, request_id: &str) -> Vec<Vec<BlockRef>> {
        self.base
            .single_type_managers
            .iter()
            .map(|manager| manager.get_blocks(request_id).unwrap_or_default())
            .collect()
    }

    fn find_longest_cache_hit(
        &mut self,
        _block_hashes: &[BlockHash],
        _max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockRef>>, usize) {
        let blocks: Vec<Vec<BlockRef>> = (0..self.num_single_type_manager)
            .map(|_| Vec::new())
            .collect();
        (blocks, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kv_cache::attention_managers::FullAttentionSpec;
    use crate::core::kv_cache::kv_cache_coordinator::KVCacheGroup;

    fn create_test_config(num_groups: usize, block_size: usize) -> KVCacheConfig {
        let groups = (0..num_groups)
            .map(|_| KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size,
                    sliding_window: None,
                }),
            })
            .collect();

        KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: groups,
        }
    }

    #[test]
    fn test_no_prefix_cache_coordinator() {
        let config = create_test_config(2, 16);
        let mut coordinator = KVCacheCoordinatorNoPrefixCache::new(config, 1024, false, false, 1);

        // Test allocate new blocks
        let blocks = coordinator.allocate_new_blocks("req1", 32, 0);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 2); // 32 / 16 = 2 blocks

        // Test get blocks
        let retrieved = coordinator.get_blocks("req1");
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].len(), 2);

        // Test common prefix blocks (should be 0 for no prefix cache)
        let common = coordinator.get_num_common_prefix_blocks("req1", 1);
        assert_eq!(common, vec![0, 0]);

        // Test cache hit (should return empty)
        let (hit_blocks, hit_len) = coordinator.find_longest_cache_hit(&[], 100);
        assert_eq!(hit_blocks.len(), 2);
        assert!(hit_blocks[0].is_empty());
        assert_eq!(hit_len, 0);

        // Test free
        coordinator.free("req1");
        let retrieved = coordinator.get_blocks("req1");
        assert!(retrieved[0].is_empty());
    }

    #[test]
    fn test_coordinator_with_computed_blocks() {
        // Test save_new_computed_blocks integration
        let config = create_test_config(2, 16);
        let mut coordinator = KVCacheCoordinatorNoPrefixCache::new(config, 1024, false, false, 1);

        // Get some blocks from pool to simulate computed blocks
        let computed_blocks1 = coordinator.base.block_pool.borrow_mut().get_new_blocks(2);
        let computed_blocks2 = coordinator.base.block_pool.borrow_mut().get_new_blocks(2);

        // Save computed blocks for a request (different blocks for each group)
        coordinator.save_new_computed_blocks("req1", &[computed_blocks1, computed_blocks2]);

        // Verify blocks are tracked
        let retrieved = coordinator.get_blocks("req1");
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0].len(), 2);

        // Allocate additional blocks
        let new_blocks = coordinator.allocate_new_blocks("req1", 64, 0);
        assert_eq!(new_blocks[0].len(), 2); // Group 1: needs 4 blocks total, has 2, gets 2 more
        assert_eq!(new_blocks[1].len(), 2); // Group 2: needs 4 blocks total, has 2, gets 2 more

        coordinator.free("req1");
    }
}