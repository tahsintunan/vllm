pub mod hybrid;
pub mod no_prefix_cache;
pub mod unitary;

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::core::kv_cache::attention_managers::{
    get_manager_for_kv_cache_spec, KVCacheSpec, SingleTypeKVCacheManager,
};
use crate::core::kv_cache::block_pool::BlockPool;
use crate::core::kv_cache::free_block_queue::{BlockHash, BlockRef};
use crate::core::types::request::Request;

#[derive(Clone)]
pub struct KVCacheGroup {
    pub kv_cache_spec: Box<dyn KVCacheSpec>,
}

#[derive(Clone)]
pub struct KVCacheConfig {
    pub num_blocks: usize,
    pub kv_cache_groups: Vec<KVCacheGroup>,
}

pub trait KVCacheCoordinator: Any {
    fn as_any(&self) -> &dyn Any;

    fn get_block_pool(&self) -> Rc<RefCell<BlockPool>>;
    fn get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[Vec<BlockRef>],
        num_encoder_tokens: usize,
    ) -> usize;

    fn save_new_computed_blocks(&mut self, request_id: &str, new_computed_blocks: &[Vec<BlockRef>]);

    fn allocate_new_blocks(
        &mut self,
        request_id: &str,
        num_tokens: usize,
        num_encoder_tokens: usize,
    ) -> Vec<Vec<BlockRef>>;

    fn cache_blocks(&mut self, request: &Request, num_computed_tokens: usize);

    fn free(&mut self, request_id: &str);

    fn get_num_common_prefix_blocks(
        &self,
        request_id: &str,
        num_running_requests: usize,
    ) -> Vec<usize>;

    fn remove_skipped_blocks(&mut self, request_id: &str, num_computed_tokens: usize);

    fn get_blocks(&self, request_id: &str) -> Vec<Vec<BlockRef>>;

    fn find_longest_cache_hit(
        &mut self,
        block_hashes: &[BlockHash],
        max_cache_hit_length: usize,
    ) -> (Vec<Vec<BlockRef>>, usize);
}

pub struct BaseKVCacheCoordinator {
    pub kv_cache_config: KVCacheConfig,
    pub max_model_len: usize,
    pub enable_caching: bool,
    pub block_pool: Rc<RefCell<BlockPool>>,
    pub use_eagle: bool,
    pub single_type_managers: Vec<Box<dyn SingleTypeKVCacheManager>>,
}

impl BaseKVCacheCoordinator {
    pub fn new(
        kv_cache_config: KVCacheConfig,
        max_model_len: usize,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: usize,
    ) -> Self {
        let block_pool = Rc::new(RefCell::new(BlockPool::new(
            kv_cache_config.num_blocks,
            enable_caching,
            enable_kv_cache_events,
        )));

        let single_type_managers = kv_cache_config
            .kv_cache_groups
            .iter()
            .enumerate()
            .map(|(i, group)| {
                get_manager_for_kv_cache_spec(
                    group.kv_cache_spec.as_ref(),
                    block_pool.clone(),
                    i as u32,
                    dcp_world_size,
                )
            })
            .collect();

        Self {
            kv_cache_config,
            max_model_len,
            enable_caching,
            block_pool,
            use_eagle,
            single_type_managers,
        }
    }
}

pub fn get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: usize,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: usize,
) -> Box<dyn KVCacheCoordinator> {
    use self::hybrid::HybridKVCacheCoordinator;
    use self::no_prefix_cache::KVCacheCoordinatorNoPrefixCache;
    use self::unitary::UnitaryKVCacheCoordinator;

    if !enable_caching {
        Box::new(KVCacheCoordinatorNoPrefixCache::new(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size,
        ))
    } else if kv_cache_config.kv_cache_groups.len() == 1 {
        Box::new(UnitaryKVCacheCoordinator::new(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
        ))
    } else {
        Box::new(HybridKVCacheCoordinator::new(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size,
        ))
    }
}

// Re-export for backward compatibility
pub use self::hybrid::{verify_and_split_kv_cache_groups, HybridKVCacheCoordinator};
pub use self::no_prefix_cache::KVCacheCoordinatorNoPrefixCache;
pub use self::unitary::UnitaryKVCacheCoordinator;
