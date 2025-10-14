// Core module that contains all the V1 core implementation

// KV Cache components
pub mod kv_cache;

// Type definitions
pub mod types;

// Scheduler module
pub mod scheduler;

// Test modules
#[cfg(test)]
mod integration_tests;

// Re-export commonly used types for convenience

// From kv_cache module - BlockPool
pub use kv_cache::block_pool::BlockPool;

// From kv_cache module
pub use kv_cache::free_block_queue::{
    BlockHash, BlockHashWithGroupId, BlockRef, FreeKVCacheBlockQueue, KVCacheBlock,
};
pub use kv_cache::kv_cache_coordinator::{
    get_kv_cache_coordinator, HybridKVCacheCoordinator, KVCacheConfig, KVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache, KVCacheGroup, UnitaryKVCacheCoordinator,
};
pub use kv_cache::kv_cache_manager::{KVCacheBlocks, KVCacheManager, PrefixCacheStats};

// From kv_cache module - attention_managers
pub use kv_cache::attention_managers::{
    AttentionType, ChunkedLocalAttentionManager, ChunkedLocalAttentionSpec, CrossAttentionManager,
    FullAttentionManager, FullAttentionSpec, KVCacheSpec, MambaManager, MambaSpec,
    SingleTypeKVCacheManager, SlidingWindowManager, SlidingWindowSpec,
};

// From types module
pub use types::engine_core_outputs::{
    EngineCoreOutput, EngineCoreOutputs, SchedulerStats, SpecDecodingStats,
};
pub use types::model_runner_output::{DraftTokenIds, KvConnectorOutput, ModelRunnerOutput};
pub use types::request::{
    EngineCoreEvent, EngineCoreEventType, FinishReason, LoraRequest, MultiModalFeatureSpec,
    PlaceholderRange, PoolingParams, Request, RequestStatus, SamplingParams, StopReason,
};
pub use types::scheduler_output::{
    CachedRequestData, KVConnectorMetadata, NewRequestData, SchedulerOutput,
};

// From scheduler module
pub use scheduler::encoder_cache_manager::{compute_encoder_budget, EncoderCacheManager};
pub use scheduler::request_queue::{
    create_request_queue, FcfsRequestQueue, PriorityRequestQueue, RequestQueue, SchedulingPolicy,
};
pub use scheduler::scheduler::{Scheduler, SchedulerConfig};
pub use scheduler::scheduler_interface::SchedulerInterface;
// pub use scheduler::scheduler_utils::{
//     calculate_chunked_prefill_tokens, check_stop, should_use_prefix_cache, EncoderBudget,
//     TokenBudget,
// };
