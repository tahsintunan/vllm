use std::cell::RefCell;
use std::rc::Rc;

use super::block_pool::BlockPool;
use super::free_block_queue::BlockRef;
use super::kv_cache_coordinator::{get_kv_cache_coordinator, KVCacheConfig, KVCacheCoordinator};

// Import Request from the request module
use crate::core::types::request::{Request, RequestStatus};

/// The allocation result of KVCacheManager
#[derive(Debug, Clone)]
pub struct KVCacheBlocks {
    /// blocks[i][j] refers to the i-th kv_cache_group and the j-th block of tokens
    pub blocks: Vec<Vec<BlockRef>>,
}

impl KVCacheBlocks {
    pub fn new(blocks: Vec<Vec<BlockRef>>) -> Self {
        Self { blocks }
    }

    /// Create an empty KVCacheBlocks with the specified number of groups
    pub fn new_empty(num_groups: usize) -> Self {
        Self {
            blocks: vec![Vec::new(); num_groups],
        }
    }

    /// Get block IDs from the blocks
    pub fn get_block_ids(&self) -> Vec<Vec<u32>> {
        self.blocks
            .iter()
            .map(|group| group.iter().map(|block| block.borrow().block_id).collect())
            .collect()
    }

    /// Get block IDs of unhashed blocks (for the first group only)
    pub fn get_unhashed_block_ids(&self) -> Vec<u32> {
        assert_eq!(self.blocks.len(), 1, "Only one group is supported");
        self.blocks[0]
            .iter()
            .filter(|block| block.borrow().hash_key.is_none())
            .map(|block| block.borrow().block_id)
            .collect()
    }

    /// Add two KVCacheBlocks instances
    pub fn add(&self, other: &KVCacheBlocks) -> KVCacheBlocks {
        assert_eq!(self.blocks.len(), other.blocks.len());
        let mut new_blocks = Vec::new();
        for (group1, group2) in self.blocks.iter().zip(other.blocks.iter()) {
            let mut combined = group1.clone();
            combined.extend(group2.clone());
            new_blocks.push(combined);
        }
        KVCacheBlocks::new(new_blocks)
    }
}

/// Prefix cache statistics
#[derive(Debug, Default, Clone)]
pub struct PrefixCacheStats {
    pub requests: usize,
    pub queries: usize,
    pub hits: usize,
    pub reset: bool,
}

/// Main KV Cache Manager
pub struct KVCacheManager {
    pub max_model_len: usize,
    pub enable_caching: bool,
    pub use_eagle: bool,
    pub log_stats: bool,
    pub block_size: Option<usize>,
    pub coordinator: Box<dyn KVCacheCoordinator>,
    pub num_kv_cache_groups: usize,
    pub block_pool: Rc<RefCell<BlockPool>>,
    pub kv_cache_config: KVCacheConfig,
    pub prefix_cache_stats: Option<PrefixCacheStats>,
}

impl KVCacheManager {
    pub fn new(
        kv_cache_config: KVCacheConfig,
        max_model_len: usize,
        enable_caching: bool,
        use_eagle: bool,
        log_stats: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
    ) -> Self {
        // Determine block size if caching is enabled
        let block_size = if enable_caching {
            // Verify all groups have the same block size
            let block_sizes: Vec<usize> = kv_cache_config
                .kv_cache_groups
                .iter()
                .map(|g| g.kv_cache_spec.block_size())
                .collect();

            let first_size = block_sizes[0];
            assert!(
                block_sizes.iter().all(|&size| size == first_size),
                "Only one block size is supported for now"
            );

            let mut size = first_size;
            if dcp_world_size > 1 {
                assert_eq!(
                    kv_cache_config.kv_cache_groups.len(),
                    1,
                    "Only one KV cache group is supported with DCP"
                );
                size *= dcp_world_size;
            }
            Some(size)
        } else {
            None
        };

        let num_kv_cache_groups = kv_cache_config.kv_cache_groups.len();

        let coordinator = get_kv_cache_coordinator(
            kv_cache_config.clone(),
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
        );

        // Get block pool from coordinator
        let block_pool = coordinator.get_block_pool();

        let prefix_cache_stats = if log_stats {
            Some(PrefixCacheStats::default())
        } else {
            None
        };

        Self {
            max_model_len,
            enable_caching,
            use_eagle,
            log_stats,
            block_size,
            coordinator,
            num_kv_cache_groups,
            block_pool,
            kv_cache_config,
            prefix_cache_stats,
        }
    }

    /// Get the KV cache usage (between 0.0 and 1.0)
    pub fn usage(&self) -> f32 {
        self.block_pool.borrow().get_usage()
    }

    /// Get and reset prefix cache stats
    pub fn make_prefix_cache_stats(&mut self) -> Option<PrefixCacheStats> {
        if !self.log_stats {
            return None;
        }
        let stats = self.prefix_cache_stats.clone();
        self.prefix_cache_stats = Some(PrefixCacheStats::default());
        stats
    }

    /// Get the computed (cached) blocks for the request
    pub fn get_computed_blocks(&mut self, request: &Request) -> (KVCacheBlocks, usize) {
        // Skip prefix caching if disabled or prompt logprobs are requested
        if !self.enable_caching || request.prompt_logprobs_enabled {
            return (self.create_empty_block_list(), 0);
        }

        // Set max cache hit length to prompt_length - 1 to ensure we recompute
        // the last token for logits
        let max_cache_hit_length = request.num_tokens().saturating_sub(1);

        let (computed_blocks, num_new_computed_tokens) = self
            .coordinator
            .find_longest_cache_hit(&request.block_hashes, max_cache_hit_length);

        if self.log_stats {
            if let Some(ref mut stats) = self.prefix_cache_stats {
                stats.requests += 1;
                stats.queries += request.num_tokens();
                stats.hits += num_new_computed_tokens;
            }
        }

        (KVCacheBlocks::new(computed_blocks), num_new_computed_tokens)
    }

    /// Allocate slots for a request with new tokens
    pub fn allocate_slots(
        &mut self,
        request: &Request,
        num_new_tokens: usize,
        num_new_computed_tokens: usize,
        new_computed_blocks: Option<&KVCacheBlocks>,
        num_lookahead_tokens: usize,
        delay_cache_blocks: bool,
        num_encoder_tokens: usize,
    ) -> Option<KVCacheBlocks> {
        if num_new_tokens == 0 {
            panic!("num_new_tokens must be greater than 0");
        }

        let new_computed_block_list = if let Some(blocks) = new_computed_blocks {
            blocks.blocks.clone()
        } else {
            vec![Vec::new(); self.num_kv_cache_groups]
        };

        // Remove blocks outside sliding window before allocation
        self.coordinator
            .remove_skipped_blocks(&request.request_id, request.num_computed_tokens);

        let num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens;
        let num_tokens_need_slot =
            (num_computed_tokens + num_new_tokens + num_lookahead_tokens).min(self.max_model_len);

        let num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            &request.request_id,
            num_tokens_need_slot,
            &new_computed_block_list,
            num_encoder_tokens,
        );

        if num_blocks_to_allocate > self.block_pool.borrow().get_num_free_blocks() {
            return None;
        }

        // Touch computed blocks to prevent eviction
        if self.enable_caching {
            self.block_pool.borrow_mut().touch(&new_computed_block_list);
        } else {
            assert!(
                new_computed_block_list.iter().all(|group| group.is_empty()),
                "Computed blocks should be empty when prefix caching is disabled"
            );
        }

        // Save the new computed blocks
        self.coordinator
            .save_new_computed_blocks(&request.request_id, &new_computed_block_list);

        // Allocate new blocks
        let new_blocks = self.coordinator.allocate_new_blocks(
            &request.request_id,
            num_tokens_need_slot,
            num_encoder_tokens,
        );

        // Skip caching if disabled or delayed
        if !self.enable_caching || delay_cache_blocks {
            return Some(KVCacheBlocks::new(new_blocks));
        }

        // Cache blocks up to the finalized tokens
        let num_tokens_to_cache = (num_computed_tokens + num_new_tokens).min(request.num_tokens());
        self.coordinator.cache_blocks(request, num_tokens_to_cache);

        Some(KVCacheBlocks::new(new_blocks))
    }

    /// Free the blocks allocated for a request
    pub fn free(&mut self, request: &Request) {
        self.coordinator.free(&request.request_id);
    }

    /// Reset prefix cache
    pub fn reset_prefix_cache(&mut self) -> bool {
        if !self.block_pool.borrow_mut().reset_prefix_cache() {
            return false;
        }
        if self.log_stats {
            if let Some(ref mut stats) = self.prefix_cache_stats {
                stats.reset = true;
            }
        }
        true
    }

    /// Get the number of common prefix blocks
    pub fn get_num_common_prefix_blocks(
        &self,
        request: &Request,
        num_running_requests: usize,
    ) -> Vec<usize> {
        assert_eq!(request.status, RequestStatus::Running);
        self.coordinator
            .get_num_common_prefix_blocks(&request.request_id, num_running_requests)
    }

    /// Get blocks for a request
    pub fn get_blocks(&self, request_id: &str) -> KVCacheBlocks {
        KVCacheBlocks::new(self.coordinator.get_blocks(request_id))
    }

    /// Get block IDs for a request
    pub fn get_block_ids(&self, request_id: &str) -> Vec<Vec<u32>> {
        self.get_blocks(request_id).get_block_ids()
    }

    /// Cache blocks for a request
    pub fn cache_blocks(&mut self, request: &Request, num_computed_tokens: usize) {
        if self.enable_caching {
            self.coordinator.cache_blocks(request, num_computed_tokens);
        }
    }

    /// Create an empty block list
    pub fn create_empty_block_list(&self) -> KVCacheBlocks {
        KVCacheBlocks::new_empty(self.num_kv_cache_groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::kv_cache::kv_cache_coordinator::KVCacheGroup;
    use crate::core::kv_cache::attention_managers::FullAttentionSpec;

    fn create_test_config() -> KVCacheConfig {
        KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        }
    }

    #[test]
    fn test_kv_cache_blocks() {
        let blocks = KVCacheBlocks::new_empty(2);
        assert_eq!(blocks.blocks.len(), 2);
        assert!(blocks.blocks[0].is_empty());

        let ids = blocks.get_block_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids[0].is_empty());
    }

    #[test]
    fn test_manager_creation() {
        let config = create_test_config();
        let manager = KVCacheManager::new(config, 1024, true, false, false, false, 1);

        assert_eq!(manager.max_model_len, 1024);
        assert!(manager.enable_caching);
        assert_eq!(manager.block_size, Some(16));
        assert_eq!(manager.num_kv_cache_groups, 1);
    }

    #[test]
    fn test_manager_allocate_slots() {
        let config = create_test_config();
        let mut manager = KVCacheManager::new(
            config, 1024, false, // disable caching for simpler test
            false, false, false, 1,
        );

        let request = Request::new_simple("req1".to_string(), 32);

        let blocks = manager.allocate_slots(
            &request, 32, // num_new_tokens
            0,  // num_new_computed_tokens
            None, 0, // num_lookahead_tokens
            false, 0, // num_encoder_tokens
        );

        assert!(blocks.is_some());
        let blocks = blocks.unwrap();
        assert_eq!(blocks.blocks[0].len(), 2); // 32 / 16 = 2 blocks
    }

    #[test]
    fn test_manager_free() {
        let config = create_test_config();
        let mut manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        let request = Request::new_simple("req1".to_string(), 32);

        // Allocate blocks
        let blocks = manager.allocate_slots(&request, 32, 0, None, 0, false, 0);
        assert!(blocks.is_some());

        // Check usage increased
        let usage_before = manager.usage();

        // Free blocks
        manager.free(&request);

        // Check usage decreased
        let usage_after = manager.usage();
        assert!(usage_after < usage_before);
    }

    #[test]
    fn test_prefix_cache_stats() {
        let config = create_test_config();
        let mut manager = KVCacheManager::new(
            config, 1024, true, false, true, // enable stats
            false, 1,
        );

        assert!(manager.prefix_cache_stats.is_some());

        // Get computed blocks should update stats
        let request = Request::new_simple("req1".to_string(), 32);
        let (_, _) = manager.get_computed_blocks(&request);

        if let Some(ref stats) = manager.prefix_cache_stats {
            assert_eq!(stats.requests, 1);
            assert_eq!(stats.queries, 32);
        }

        // Make stats should reset them
        let stats = manager.make_prefix_cache_stats();
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().requests, 1);

        // New stats should be reset
        if let Some(ref stats) = manager.prefix_cache_stats {
            assert_eq!(stats.requests, 0);
        }
    }

    #[test]
    fn test_reset_prefix_cache() {
        let config = create_test_config();
        let mut manager = KVCacheManager::new(config, 1024, true, false, true, false, 1);

        // Should succeed when no blocks are in use
        assert!(manager.reset_prefix_cache());

        // Should update stats
        if let Some(ref stats) = manager.prefix_cache_stats {
            assert!(stats.reset);
        }
    }

    #[test]
    fn test_get_blocks_and_ids() {
        let config = create_test_config();
        let mut manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        let request = Request::new_simple("req1".to_string(), 32);

        // Allocate blocks
        manager.allocate_slots(&request, 32, 0, None, 0, false, 0);

        // Get blocks
        let blocks = manager.get_blocks("req1");
        assert_eq!(blocks.blocks[0].len(), 2);

        // Get block IDs
        let ids = manager.get_block_ids("req1");
        assert_eq!(ids[0].len(), 2);
    }

    #[test]
    fn test_common_prefix_blocks() {
        let config = create_test_config();
        let mut manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        let mut request = Request::new_simple("req1".to_string(), 32);
        request.status = RequestStatus::Running;

        // Allocate blocks
        manager.allocate_slots(&request, 32, 0, None, 0, false, 0);

        // Get common prefix blocks
        let common = manager.get_num_common_prefix_blocks(&request, 1);
        assert_eq!(common.len(), 1);
    }

    #[test]
    fn test_manager_coordinator_integration_allocation() {
        // Test that KVCacheManager properly delegates to coordinator
        let config = create_test_config();
        let mut manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        let request1 = Request::new_simple("req1".to_string(), 48);
        let request2 = Request::new_simple("req2".to_string(), 32);

        // Allocate for multiple requests through manager
        let blocks1 = manager.allocate_slots(&request1, 48, 0, None, 0, false, 0);
        let blocks2 = manager.allocate_slots(&request2, 32, 0, None, 0, false, 0);

        assert!(blocks1.is_some());
        assert!(blocks2.is_some());

        // Verify coordinator is tracking both requests
        let retrieved1 = manager.get_blocks("req1");
        let retrieved2 = manager.get_blocks("req2");

        assert_eq!(retrieved1.blocks[0].len(), 3); // 48/16 = 3
        assert_eq!(retrieved2.blocks[0].len(), 2); // 32/16 = 2

        // Free one request and verify
        manager.free(&request1);
        let after_free = manager.get_blocks("req1");
        assert!(after_free.blocks[0].is_empty());

        // Request 2 should still have blocks
        let still_there = manager.get_blocks("req2");
        assert_eq!(still_there.blocks[0].len(), 2);
    }

    #[test]
    fn test_manager_with_prefix_caching() {
        // Test prefix caching through manager->coordinator integration
        let config = create_test_config();
        let mut manager = KVCacheManager::new(
            config, 1024, true, // enable caching
            false, true, // enable stats
            false, 1,
        );

        // Use 32 tokens (2 full blocks) for cleaner testing
        let mut request = Request::new_simple("req1".to_string(), 32);
        request.block_hashes = vec![[1u8; 32], [2u8; 32]];

        // First allocation
        let blocks = manager.allocate_slots(&request, 32, 0, None, 0, false, 0);
        assert!(blocks.is_some());

        // Cache the blocks
        manager.cache_blocks(&request, 32);

        // Free the request
        manager.free(&request);

        // New request with same block hashes should get cache hit
        let mut request2 = Request::new_simple("req2".to_string(), 32);
        request2.block_hashes = vec![[1u8; 32], [2u8; 32]];

        let (computed_blocks, hit_tokens) = manager.get_computed_blocks(&request2);

        // With max_cache_hit_length=31 (32-1), we check floor(31/16)=1 block
        // So we get 1 block * 16 tokens = 16 tokens as cache hit
        assert_eq!(hit_tokens, 16);
        assert_eq!(computed_blocks.blocks[0].len(), 1); // floor(31/16) = 1 block

        // Stats should reflect the cache hit
        if let Some(ref stats) = manager.prefix_cache_stats {
            assert_eq!(stats.hits, 16);
            assert_eq!(stats.queries, 32);
            assert_eq!(stats.requests, 1);
        }
    }

    #[test]
    fn test_manager_sliding_window_integration() {
        // Test sliding window through manager->coordinator
        use crate::core::kv_cache::attention_managers::SlidingWindowSpec;

        let config = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(SlidingWindowSpec {
                    block_size: 16,
                    sliding_window: 32,
                }),
            }],
        };

        let mut manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        let mut request = Request::new_simple("req1".to_string(), 80);
        request.num_computed_tokens = 64;

        // Allocate blocks
        let blocks = manager.allocate_slots(&request, 16, 0, None, 0, false, 0);
        assert!(blocks.is_some());

        // Should have removed blocks outside sliding window
        // The coordinator's remove_skipped_blocks should have been called
        let all_blocks = manager.get_blocks("req1");
        assert_eq!(all_blocks.blocks[0].len(), 5); // 80/16 = 5 total
    }

    #[test]
    fn test_manager_coordinator_pool_exhaustion() {
        // Test that manager properly handles pool exhaustion from coordinator
        let config = KVCacheConfig {
            num_blocks: 5, // Very small pool
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        };

        let mut manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        // First allocation should succeed (uses 4 blocks)
        let request1 = Request::new_simple("req1".to_string(), 64);
        let blocks1 = manager.allocate_slots(&request1, 64, 0, None, 0, false, 0);
        assert!(blocks1.is_some());

        // Second allocation should fail - not enough blocks
        let request2 = Request::new_simple("req2".to_string(), 32);
        let blocks2 = manager.allocate_slots(&request2, 32, 0, None, 0, false, 0);
        assert!(blocks2.is_none()); // Should return None when not enough blocks

        // Free first request
        manager.free(&request1);

        // Now second request should succeed
        let blocks2_retry = manager.allocate_slots(&request2, 32, 0, None, 0, false, 0);
        assert!(blocks2_retry.is_some());
    }

    #[test]
    fn test_manager_different_coordinator_types() {
        // Test that manager works with different coordinator types

        // Test with NoPrefixCache coordinator
        let config1 = create_test_config();
        let manager1 = KVCacheManager::new(
            config1, 1024, false, // disable caching -> NoPrefixCache coordinator
            false, false, false, 1,
        );
        assert_eq!(manager1.block_size, None);

        // Test with Unitary coordinator
        let config2 = create_test_config();
        let manager2 = KVCacheManager::new(
            config2, 1024, true, // enable caching with 1 group -> Unitary coordinator
            false, false, false, 1,
        );
        assert_eq!(manager2.block_size, Some(16));

        // Test with Hybrid coordinator
        use crate::core::kv_cache::attention_managers::SlidingWindowSpec;
        let config3 = KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: vec![
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
            ],
        };

        let manager3 = KVCacheManager::new(
            config3, 1024, true, // enable caching with 2 groups -> Hybrid coordinator
            false, false, false, 1,
        );
        assert_eq!(manager3.num_kv_cache_groups, 2);
    }

    #[test]
    fn test_manager_save_computed_blocks() {
        // Test save_new_computed_blocks through manager->coordinator
        let config = create_test_config();
        let mut manager = KVCacheManager::new(config, 1024, true, false, false, false, 1);

        // Get some blocks from pool to simulate computed blocks (2 blocks = 32 tokens)
        let computed_blocks_raw = manager.block_pool.borrow_mut().get_new_blocks(2);
        let computed_blocks = KVCacheBlocks::new(vec![computed_blocks_raw]);

        let mut request = Request::new_simple("req1".to_string(), 64);
        request.num_computed_tokens = 0; // Starting fresh, no tokens computed yet
                                         // Add block hashes for 64 tokens (4 blocks)
        request.block_hashes = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];

        // Allocate slots: 32 tokens from computed blocks + 32 new tokens
        let blocks = manager.allocate_slots(
            &request,
            32, // num_new_tokens (tokens 33-64)
            32, // num_new_computed_tokens (tokens 1-32 from cache)
            Some(&computed_blocks),
            0,
            false,
            0,
        );

        assert!(blocks.is_some());
        let all_blocks = manager.get_blocks("req1");

        // Should have computed blocks (2) + new blocks (2 more for remaining 32 tokens)
        assert_eq!(all_blocks.blocks[0].len(), 4);
    }
}
