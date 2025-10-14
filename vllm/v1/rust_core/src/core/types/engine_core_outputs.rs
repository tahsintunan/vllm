use std::collections::HashMap;

use super::request::{EngineCoreEvent, FinishReason, StopReason};

/// Container for outputs sent to engine clients
///
/// This is the main output structure returned from update_from_output()
/// containing all outputs for requests from a specific client.
pub struct EngineCoreOutputs {
    /// List of outputs for individual requests
    pub outputs: Vec<EngineCoreOutput>,

    /// Scheduler statistics for this iteration
    pub scheduler_stats: Option<SchedulerStats>,

    /// IDs of requests that finished in this iteration
    pub finished_requests: Vec<String>,

    /// Whether this wave of data parallelism is complete
    pub wave_complete: bool,
}

/// Output for a single request
///
/// Contains the generated tokens and associated metadata for one request
/// after a single scheduling step.
pub struct EngineCoreOutput {
    /// Unique identifier for this request
    pub request_id: String,

    /// Newly generated token IDs in this step
    pub new_token_ids: Vec<i32>,

    /// Reason why the request finished (if it did)
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities for the new tokens
    pub new_logprobs: Option<Vec<f32>>,

    /// Log probabilities for prompt tokens
    pub new_prompt_logprobs_tensors: Option<Vec<f32>>,

    /// Pooling output for embedding models
    pub pooling_output: Option<Vec<f32>>,

    /// Reason why generation stopped (for partial completions)
    pub stop_reason: Option<StopReason>,

    /// Events that occurred during processing
    pub events: Vec<EngineCoreEvent>,

    /// Parameters for KV cache transfer between nodes
    pub kv_transfer_params: Option<HashMap<String, serde_json::Value>>,

    /// Trace headers for debugging/monitoring
    pub trace_headers: Option<HashMap<String, String>>,

    /// Number of cached tokens used for this request
    pub num_cached_tokens: i32,
}

/// Statistics about the scheduler state
///
/// Provides insight into the scheduler's current workload and resource usage.
pub struct SchedulerStats {
    /// Number of requests currently being processed
    pub num_running_reqs: usize,

    /// Number of requests waiting to be scheduled
    pub num_waiting_reqs: usize,

    /// Percentage of KV cache currently in use
    pub kv_cache_usage: f32,

    /// Statistics about prefix cache hits/misses
    pub prefix_cache_stats: crate::core::PrefixCacheStats,

    /// Statistics for speculative decoding (if enabled)
    pub spec_decoding_stats: Option<SpecDecodingStats>,

    /// Number of requests that had corrupted state
    pub num_corrupted_reqs: usize,
}

/// Statistics for speculative decoding
pub struct SpecDecodingStats {
    /// Number of draft tokens generated
    pub num_draft_tokens: usize,

    /// Number of draft tokens that were accepted
    pub num_accepted_tokens: usize,
}