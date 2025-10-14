use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::{BaseKVCacheCoordinator, KVCacheConfig, KVCacheCoordinator};
use crate::core::kv_cache::attention_managers::KVCacheSpec;
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};
use crate::core::types::request::Request;

pub struct UnitaryKVCacheCoordinator {
    base: BaseKVCacheCoordinator,
    kv_cache_spec: Box<dyn KVCacheSpec>,
    block_size: usize,
    dcp_world_size: usize,
}

impl UnitaryKVCacheCoordinator {
    pub fn new(
        kv_cache_config: KVCacheConfig,
        max_model_len: usize,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
    ) -> Self {
        assert_eq!(
            kv_cache_config.kv_cache_groups.len(),
            1,
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        );

        let kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec.clone();
        let mut block_size = kv_cache_spec.block_size();
        if dcp_world_size > 1 {
            block_size *= dcp_world_size;
        }

        let base = BaseKVCacheCoordinator::new(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
        );

        Self {
            base,
            kv_cache_spec,
            block_size,
            dcp_world_size,
        }
    }
}

impl KVCacheCoordinator for UnitaryKVCacheCoordinator {
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
        request_id: &str,
        num_running_requests: usize,
    ) -> Vec<usize> {
        self.base
            .single_type_managers
            .iter()
            .map(|manager| manager.get_num_common_prefix_blocks(request_id, num_running_requests))
            .collect()
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
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockRef>>, usize) {
        let hit_blocks = self.base.single_type_managers[0].find_longest_cache_hit(
            block_hashes,
            max_cache_hit_length,
            &[0],
            self.base.block_pool.clone(),
            self.kv_cache_spec.clone(),
            self.base.use_eagle,
            self.dcp_world_size,
        );
        let hit_length = hit_blocks[0].len() * self.block_size;
        (hit_blocks, hit_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kv_cache::attention_managers::{FullAttentionSpec, SlidingWindowSpec};
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
    fn test_unitary_coordinator() {
        let config = create_test_config(1, 16);
        let mut coordinator = UnitaryKVCacheCoordinator::new(config, 1024, false, true, false, 1);

        // Test allocate
        let blocks = coordinator.allocate_new_blocks("req1", 48, 0);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].len(), 3); // 48 / 16 = 3 blocks

        // Test get blocks
        let retrieved = coordinator.get_blocks("req1");
        assert_eq!(retrieved[0].len(), 3);
    }

    #[test]
    fn test_coordinator_block_pool_integration() {
        // Test that coordinator properly manages BlockPool resources
        let config = create_test_config(1, 16);
        let mut coordinator = UnitaryKVCacheCoordinator::new(config, 1024, false, false, false, 1);

        // Check initial pool state
        let initial_free = coordinator.base.block_pool.borrow().get_num_free_blocks();
        assert_eq!(initial_free, 99); // 100 - 1 null block

        // Allocate blocks for multiple requests
        let blocks1 = coordinator.allocate_new_blocks("req1", 32, 0);
        assert_eq!(blocks1[0].len(), 2);
        assert_eq!(
            coordinator.base.block_pool.borrow().get_num_free_blocks(),
            97
        );

        let blocks2 = coordinator.allocate_new_blocks("req2", 48, 0);
        assert_eq!(blocks2[0].len(), 3);
        assert_eq!(
            coordinator.base.block_pool.borrow().get_num_free_blocks(),
            94
        );

        // Free one request
        coordinator.free("req1");
        assert_eq!(
            coordinator.base.block_pool.borrow().get_num_free_blocks(),
            96
        );

        // Free second request
        coordinator.free("req2");
        assert_eq!(
            coordinator.base.block_pool.borrow().get_num_free_blocks(),
            99
        );
    }

    #[test]
    fn test_coordinator_caching_integration() {
        // Test caching functionality through coordinator
        let config = create_test_config(1, 16);
        let mut coordinator = UnitaryKVCacheCoordinator::new(
            config, 1024, false, true, // enable_caching
            false, 1,
        );

        // Allocate and cache blocks
        let blocks = coordinator.allocate_new_blocks("req1", 48, 0);
        assert_eq!(blocks[0].len(), 3);

        // Create a request with block hashes
        let mut request = Request::new_simple("req1".to_string(), 48);
        request.block_hashes = vec![[1u8; 32], [2u8; 32], [3u8; 32]];

        // Cache the blocks
        coordinator.cache_blocks(&request, 48);

        // Test cache hit
        let (hit_blocks, hit_len) = coordinator.find_longest_cache_hit(&request.block_hashes, 48);
        assert_eq!(hit_blocks[0].len(), 3);
        assert_eq!(hit_len, 48);

        coordinator.free("req1");
    }

    #[test]
    fn test_sliding_window_remove_blocks() {
        // Test remove_skipped_blocks for sliding window
        let groups = vec![KVCacheGroup {
            kv_cache_spec: Box::new(SlidingWindowSpec {
                block_size: 16,
                sliding_window: 32,
            }),
        }];

        let config = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: groups,
        };

        let mut coordinator = UnitaryKVCacheCoordinator::new(config, 1024, false, false, false, 1);

        // Allocate blocks
        let blocks = coordinator.allocate_new_blocks("req1", 80, 0);
        assert_eq!(blocks[0].len(), 5); // 80 / 16 = 5

        let initial_free = coordinator.base.block_pool.borrow().get_num_free_blocks();

        // Remove blocks outside sliding window
        coordinator.remove_skipped_blocks("req1", 64);

        // Should have freed some blocks back to pool
        assert!(coordinator.base.block_pool.borrow().get_num_free_blocks() > initial_free);

        coordinator.free("req1");
    }

    #[test]
    #[should_panic(expected = "Cannot get")]
    fn test_coordinator_pool_exhaustion() {
        // Test that coordinator properly handles pool exhaustion
        let config = KVCacheConfig {
            num_blocks: 5, // Very small pool
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        };

        let mut coordinator = UnitaryKVCacheCoordinator::new(config, 1024, false, false, false, 1);

        // This should succeed (4 blocks available)
        coordinator.allocate_new_blocks("req1", 64, 0); // 4 blocks

        // This should panic - no blocks left
        coordinator.allocate_new_blocks("req2", 16, 0); // Would need 1 more block
    }
}