use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use super::{BaseKVCacheCoordinator, KVCacheConfig, KVCacheCoordinator};
use crate::core::kv_cache::attention_managers::{AttentionType, KVCacheSpec};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};
use crate::core::types::request::Request;

pub struct HybridKVCacheCoordinator {
    base: BaseKVCacheCoordinator,
    full_attention_group_ids: Vec<u32>,
    other_group_ids: Vec<u32>,
    full_attention_spec: Box<dyn KVCacheSpec>,
    other_spec: Box<dyn KVCacheSpec>,
    full_attention_block_size: usize,
    other_block_size: usize,
    full_attn_first: bool,
}

impl HybridKVCacheCoordinator {
    pub fn new(
        kv_cache_config: KVCacheConfig,
        max_model_len: usize,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
    ) -> Self {
        assert_eq!(dcp_world_size, 1, "DCP not support hybrid attn now.");

        let base = BaseKVCacheCoordinator::new(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
        );

        let (
            full_attention_group_ids,
            other_group_ids,
            full_attention_spec,
            other_spec,
            full_attn_first,
        ) = verify_and_split_kv_cache_groups(&base.kv_cache_config, enable_caching);

        let full_attention_block_size = full_attention_spec.block_size();
        let other_block_size = other_spec.block_size();

        Self {
            base,
            full_attention_group_ids,
            other_group_ids,
            full_attention_spec,
            other_spec,
            full_attention_block_size,
            other_block_size,
            full_attn_first,
        }
    }
}

impl KVCacheCoordinator for HybridKVCacheCoordinator {
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
        // First, find the longest cache hit for full attention
        let full_attn_manager_idx = self.full_attention_group_ids[0] as usize;
        let mut hit_blocks_full_attn = self.base.single_type_managers[full_attn_manager_idx]
            .find_longest_cache_hit(
                block_hashes,
                max_cache_hit_length,
                &self.full_attention_group_ids,
                self.base.block_pool.clone(),
                self.full_attention_spec.clone(),
                self.base.use_eagle,
                1, // dcp_world_size
            );

        let mut hit_length = hit_blocks_full_attn[0].len() * self.full_attention_block_size;

        // Find cache hit for other attention within the full attention cache hit
        let other_manager_idx = self.other_group_ids[0] as usize;
        let hit_blocks_other_attn = self.base.single_type_managers[other_manager_idx]
            .find_longest_cache_hit(
                block_hashes,
                hit_length,
                &self.other_group_ids,
                self.base.block_pool.clone(),
                self.other_spec.clone(),
                self.base.use_eagle,
                1, // dcp_world_size
            );

        hit_length = hit_blocks_other_attn[0].len() * self.other_block_size;

        // Ensure hit_length is a multiple of full attention block size
        assert_eq!(hit_length % self.full_attention_block_size, 0);

        // Truncate full attention blocks to match other attention hit length
        for group_hit_blocks in hit_blocks_full_attn.iter_mut() {
            group_hit_blocks.truncate(hit_length / self.full_attention_block_size);
        }

        // Merge hit blocks based on ordering
        let hit_blocks = if self.full_attn_first {
            [hit_blocks_full_attn, hit_blocks_other_attn].concat()
        } else {
            [hit_blocks_other_attn, hit_blocks_full_attn].concat()
        };

        (hit_blocks, hit_length)
    }
}

pub fn verify_and_split_kv_cache_groups(
    kv_cache_config: &KVCacheConfig,
    enable_caching: bool,
) -> (
    Vec<u32>,
    Vec<u32>,
    Box<dyn KVCacheSpec>,
    Box<dyn KVCacheSpec>,
    bool,
) {
    let mut full_attention_spec: Option<Box<dyn KVCacheSpec>> = None;
    let mut other_spec: Option<Box<dyn KVCacheSpec>> = None;
    let mut full_attention_group_ids: Vec<u32> = Vec::new();
    let mut other_group_ids: Vec<u32> = Vec::new();

    for (i, group) in kv_cache_config.kv_cache_groups.iter().enumerate() {
        if group.kv_cache_spec.spec_type() == AttentionType::FullAttention {
            if full_attention_spec.is_none() {
                full_attention_spec = Some(group.kv_cache_spec.clone());
            }
            full_attention_group_ids.push(i as u32);
        } else {
            if other_spec.is_none() {
                other_spec = Some(group.kv_cache_spec.clone());
            }
            other_group_ids.push(i as u32);
        }
    }

    let full_attention_spec = full_attention_spec
        .expect("HybridKVCacheCoordinator assumes exactly one type of full attention groups");
    let other_spec =
        other_spec.expect("HybridKVCacheCoordinator assumes exactly one type of other groups");

    let full_attention_block_size = full_attention_spec.block_size();
    let other_block_size = other_spec.block_size();

    if enable_caching {
        assert_eq!(
            other_block_size % full_attention_block_size,
            0,
            "KVCacheCoordinator assumes the block_size of full attention layers is divisible by other layers"
        );
    }

    let full_attn_first = if full_attention_group_ids.iter().max().unwrap()
        < other_group_ids.iter().min().unwrap()
    {
        true
    } else if other_group_ids.iter().max().unwrap() < full_attention_group_ids.iter().min().unwrap()
    {
        false
    } else {
        panic!(
            "HybridKVCacheCoordinator assumes the full attention group ids and other attention \
             group ids do not interleave"
        );
    };

    (
        full_attention_group_ids,
        other_group_ids,
        full_attention_spec,
        other_spec,
        full_attn_first,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kv_cache::attention_managers::{FullAttentionSpec, SlidingWindowSpec};
    use crate::core::kv_cache::kv_cache_coordinator::get_kv_cache_coordinator;
    use crate::core::kv_cache::kv_cache_coordinator::KVCacheGroup;

    #[test]
    fn test_hybrid_coordinator() {
        // Create config with full attention and sliding window
        let groups = vec![
            KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            },
            KVCacheGroup {
                kv_cache_spec: Box::new(SlidingWindowSpec {
                    block_size: 32,
                    sliding_window: 128,
                }),
            },
        ];

        let config = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: groups,
        };

        let mut coordinator = HybridKVCacheCoordinator::new(config, 1024, false, true, false, 1);

        // Test allocate for hybrid
        let blocks = coordinator.allocate_new_blocks("req1", 64, 0);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 4); // 64 / 16 = 4 blocks for full attention
        assert_eq!(blocks[1].len(), 2); // 64 / 32 = 2 blocks for sliding window
    }

    #[test]
    fn test_hybrid_coordinator_multi_manager() {
        // Test that hybrid coordinator properly manages multiple managers
        let groups = vec![
            KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            },
            KVCacheGroup {
                kv_cache_spec: Box::new(SlidingWindowSpec {
                    block_size: 16,
                    sliding_window: 64,
                }),
            },
        ];

        let config = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: groups,
        };

        let mut coordinator = HybridKVCacheCoordinator::new(config, 1024, false, false, false, 1);

        // Allocate blocks - should allocate for both managers
        let blocks = coordinator.allocate_new_blocks("req1", 48, 0);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 3); // Full attention: 48/16 = 3
        assert_eq!(blocks[1].len(), 3); // Sliding window: 48/16 = 3

        // Remove skipped blocks (only affects sliding window manager)
        let initial_free = coordinator.base.block_pool.borrow().get_num_free_blocks();
        coordinator.remove_skipped_blocks("req1", 80);

        // Sliding window manager should have freed some blocks
        let after_remove = coordinator.base.block_pool.borrow().get_num_free_blocks();
        assert!(after_remove > initial_free);

        // Get common prefix blocks
        let common = coordinator.get_num_common_prefix_blocks("req1", 1);
        assert_eq!(common.len(), 2);

        coordinator.free("req1");
    }

    #[test]
    fn test_factory_function() {
        // Test no prefix cache
        let config = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        };
        let _coordinator = get_kv_cache_coordinator(
            config, 1024, false, false, // enable_caching = false
            false, 1,
        );
        // Should create NoPrefixCache coordinator

        // Test unitary
        let config = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        };
        let _coordinator = get_kv_cache_coordinator(
            config, 1024, false, true, // enable_caching = true
            false, 1,
        );
        // Should create Unitary coordinator

        // Test hybrid - create with different attention types
        let groups = vec![
            KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            },
            KVCacheGroup {
                kv_cache_spec: Box::new(SlidingWindowSpec {
                    block_size: 16,
                    sliding_window: 128,
                }),
            },
        ];

        let config = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: groups,
        };

        let _coordinator = get_kv_cache_coordinator(config, 1024, false, true, false, 1);
        // Should create Hybrid coordinator
    }
}
