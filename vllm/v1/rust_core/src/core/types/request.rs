use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use serde_json;

/// Reason a request finished
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Abort => "abort",
        }
    }
}

/// Status of a request
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestStatus {
    Waiting = 1,
    WaitingForFsm = 2,
    WaitingForRemoteKvs = 3,
    Running = 4,
    Preempted = 5,
    // Note: anything after PREEMPTED will be considered as a finished status
    FinishedStopped = 6,
    FinishedLengthCapped = 7,
    FinishedAborted = 8,
    FinishedIgnored = 9,
}

impl RequestStatus {
    pub fn is_finished(&self) -> bool {
        *self > RequestStatus::Preempted
    }

    pub fn get_finished_reason(&self) -> Option<FinishReason> {
        match self {
            RequestStatus::FinishedStopped => Some(FinishReason::Stop),
            RequestStatus::FinishedLengthCapped => Some(FinishReason::Length),
            RequestStatus::FinishedAborted => Some(FinishReason::Abort),
            RequestStatus::FinishedIgnored => Some(FinishReason::Length), // Same as Python
            _ => None,
        }
    }
}

/// Engine core event type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineCoreEventType {
    Queued = 1,
    Scheduled = 2,
    Preempted = 3,
}

/// Timestamped engine core event
/// The timestamp is a monotonic timestamp and is used by the engine
/// frontend to calculate intervals between engine core events. These
/// timestamps should not be compared with timestamps from other processes.
#[derive(Debug, Clone)]
pub struct EngineCoreEvent {
    pub event_type: EngineCoreEventType,
    pub timestamp: f64,
}

// Global start time for monotonic clock - initialized once
static START_TIME: OnceLock<Instant> = OnceLock::new();

/// Get monotonic time in seconds as f64, similar to Python's time.monotonic()
fn monotonic() -> f64 {
    let start = START_TIME.get_or_init(|| Instant::now());
    Instant::now().duration_since(*start).as_secs_f64()
}

impl EngineCoreEvent {
    pub fn new(event_type: EngineCoreEventType, timestamp: Option<f64>) -> Self {
        let timestamp = timestamp.unwrap_or_else(monotonic);
        Self {
            event_type,
            timestamp,
        }
    }
}

/// Sampling parameters for text generation
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate per output sequence.
    pub max_tokens: Option<u32>,
    /// Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.
    pub ignore_eos: bool,
    /// Token IDs that stop the generation when they are generated.
    pub stop_token_ids: Option<Vec<i32>>,

    // Fields not used by core scheduler but needed for pass-through
    pub n: i32,
    pub best_of: Option<i32>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub seed: Option<i64>,
    pub stop: Option<Vec<String>>,
    pub min_tokens: i32,
    pub logprobs: Option<i32>,
    pub prompt_logprobs: Option<i32>,
    pub detokenize: bool,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub include_stop_str_in_output: bool,
    pub output_kind: i32,  // RequestOutputKind enum as int
    pub output_text_buffer_length: i32,
    pub truncate_prompt_tokens: Option<i32>,
    pub logit_bias: Option<HashMap<i32, f32>>,
    pub allowed_token_ids: Option<Vec<i32>>,
    pub bad_words: Option<Vec<String>>,

    // Structured output fields
    pub json: Option<String>,
    pub regex: Option<String>,
    pub choice: Option<Vec<String>>,
    pub grammar: Option<String>,
    pub json_object: Option<bool>,
    pub disable_fallback: bool,
    pub disable_any_whitespace: bool,
    pub disable_additional_properties: bool,
    pub whitespace_pattern: Option<String>,
    pub structural_tag: Option<String>,

    // Additional pass-through fields for custom processing
    pub logits_processors: Option<serde_json::Value>,  // Arbitrary logits processing functions
    pub extra_args: Option<HashMap<String, serde_json::Value>>,  // Arbitrary additional args for plugins

    // Note: Omitting internal/private fields like _backend, _real_n, _all_stop_token_ids, etc.
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: Some(16),
            ignore_eos: false,
            stop_token_ids: None,
            n: 1,
            best_of: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
            stop: None,
            min_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
            detokenize: true,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            include_stop_str_in_output: false,
            output_kind: 0,  // CUMULATIVE
            output_text_buffer_length: 0,
            truncate_prompt_tokens: None,
            logit_bias: None,
            allowed_token_ids: None,
            bad_words: None,
            json: None,
            regex: None,
            choice: None,
            grammar: None,
            json_object: None,
            disable_fallback: false,
            disable_any_whitespace: false,
            disable_additional_properties: false,
            whitespace_pattern: None,
            structural_tag: None,
            logits_processors: None,
            extra_args: None,
        }
    }
}

/// Pooling parameters for pooling models
#[derive(Debug, Clone)]
pub struct PoolingParams {
    /// Controls prompt truncation. -1 for model's default, k>0 to keep last k tokens, None to disable
    pub truncate_prompt_tokens: Option<i32>,

    // Fields not used by core scheduler but needed for pass-through
    pub dimensions: Option<i32>,
    pub normalize: Option<bool>,
    pub activation: Option<bool>,
    pub softmax: Option<bool>,
    pub step_tag_id: Option<i32>,  // For reward models
    pub returned_token_ids: Option<Vec<i32>>,  // Specifies which token IDs to return
    pub task: Option<String>,  // Task type: "embed", "classify", "score", "encode"
    pub requires_token_ids: bool,  // Internal flag for token ID requirements
    pub extra_kwargs: Option<HashMap<String, serde_json::Value>>,  // Custom parameters
    pub output_kind: i32,  // RequestOutputKind enum, defaults to FINAL_ONLY (1)
}

impl Default for PoolingParams {
    fn default() -> Self {
        Self {
            truncate_prompt_tokens: None,
            dimensions: None,
            normalize: None,
            activation: None,
            softmax: None,
            step_tag_id: None,
            returned_token_ids: None,
            task: None,
            requires_token_ids: false,
            extra_kwargs: None,
            output_kind: 1,  // FINAL_ONLY
        }
    }
}

/// LoRA request information
#[derive(Debug, Clone)]
pub struct LoraRequest {
    /// Globally unique ID for the adapter (used by scheduler)
    pub lora_int_id: i32,

    // Fields not used by core scheduler but needed for pass-through
    pub lora_name: String,
    pub lora_path: String,
    pub lora_local_path: Option<String>,
    pub long_lora_max_len: Option<i32>,
    pub base_model_name: Option<String>,
    pub tensorizer_config_dict: Option<HashMap<String, serde_json::Value>>,  // Configuration for tensorizer
}

/// Placeholder location information for multi-modal data
#[derive(Debug, Clone)]
pub struct PlaceholderRange {
    pub offset: usize,
    pub length: usize,
}

/// Multi-modal feature specification
#[derive(Debug, Clone)]
pub struct MultiModalFeatureSpec {
    /// mm_hash or uuid for caching encoder outputs
    pub identifier: String,
    /// Position in the sequence
    pub mm_position: PlaceholderRange,

    // Fields not used by core scheduler but needed for pass-through
    pub modality: String,  // "image", "audio", "video", etc.
    // Note: 'data' field would be an opaque pointer or serialized bytes in actual FFI
}

// Use BlockHash from kv_cache module
use crate::core::kv_cache::free_block_queue::BlockHash;

/// Main Request struct
#[derive(Debug, Clone)]
pub struct Request {
    pub request_id: String,
    pub prompt_token_ids: Vec<i32>,
    pub sampling_params: Option<SamplingParams>,
    pub pooling_params: Option<PoolingParams>,
    pub eos_token_id: Option<i32>,
    pub client_index: usize,
    pub arrival_time: f64,
    pub mm_features: Vec<MultiModalFeatureSpec>,
    pub lora_request: Option<LoraRequest>,
    /// Whether structured output (FSM-guided generation) is enabled
    pub use_structured_output: bool,
    pub priority: i32,

    // State fields
    pub status: RequestStatus,
    pub events: Vec<EngineCoreEvent>,
    pub stop_reason: Option<StopReason>,

    // Token management
    pub max_tokens: u32,
    pub num_prompt_tokens: usize,
    pub output_token_ids: Vec<i32>,
    pub all_token_ids: Vec<i32>,
    pub num_output_placeholders: usize,
    pub spec_token_ids: Vec<i32>,
    pub num_computed_tokens: usize,

    // Multi-modal related
    pub has_encoder_inputs: bool,

    // Caching related
    pub num_cached_tokens: i32,
    pub block_hashes: Vec<BlockHash>,

    // Whether prompt logprobs are enabled
    pub prompt_logprobs_enabled: bool,
}

/// Stop reason can be either a token ID or a string
#[derive(Debug, Clone)]
pub enum StopReason {
    TokenId(i32),
    String(String),
}

impl Request {
    pub fn new(
        request_id: String,
        prompt_token_ids: Vec<i32>,
        sampling_params: Option<SamplingParams>,
        pooling_params: Option<PoolingParams>,
        eos_token_id: Option<i32>,
        client_index: usize,
        arrival_time: Option<f64>,
        mm_features: Option<Vec<MultiModalFeatureSpec>>,
        lora_request: Option<LoraRequest>,
        use_structured_output: bool,
        priority: i32,
    ) -> Self {
        let arrival_time = arrival_time.unwrap_or_else(monotonic);

        let mm_features = mm_features.unwrap_or_default();
        let has_encoder_inputs = !mm_features.is_empty();

        let num_prompt_tokens = prompt_token_ids.len();
        let all_token_ids = prompt_token_ids.clone();

        // Determine status and max_tokens
        let mut status = RequestStatus::Waiting;
        let max_tokens;

        if let Some(ref _pooling) = pooling_params {
            // Pooling models
            max_tokens = 1;
        } else if let Some(ref sampling) = sampling_params {
            // Generative models
            // TODO: Python asserts max_tokens is not None, we provide a default of 16.
            // Should we match Python's stricter behavior and require it to be set?
            max_tokens = sampling.max_tokens.unwrap_or(16);
            if use_structured_output {
                status = RequestStatus::WaitingForFsm;
            }
        } else {
            panic!("sampling_params and pooling_params can't both be unset");
        }

        Self {
            request_id,
            prompt_token_ids: prompt_token_ids.clone(),
            sampling_params,
            pooling_params,
            eos_token_id,
            client_index,
            arrival_time,
            mm_features,
            lora_request,
            use_structured_output,
            priority,
            status,
            events: Vec::new(),
            stop_reason: None,
            max_tokens,
            num_prompt_tokens,
            output_token_ids: Vec::new(),
            all_token_ids,
            num_output_placeholders: 0,
            spec_token_ids: Vec::new(),
            num_computed_tokens: 0,
            has_encoder_inputs,
            num_cached_tokens: -1,
            block_hashes: Vec::new(),
            prompt_logprobs_enabled: false,
        }
    }

    pub fn append_output_token_ids(&mut self, token_ids: &[i32]) {
        self.output_token_ids.extend_from_slice(token_ids);
        self.all_token_ids.extend_from_slice(token_ids);

        // In Python, this would also update block_hashes if block_hasher is set
        // We'll handle that when we implement block hashing
    }

    pub fn append_output_token_id(&mut self, token_id: i32) {
        self.output_token_ids.push(token_id);
        self.all_token_ids.push(token_id);
    }

    pub fn num_tokens(&self) -> usize {
        self.all_token_ids.len()
    }

    pub fn num_tokens_with_spec(&self) -> usize {
        self.all_token_ids.len() + self.spec_token_ids.len()
    }

    pub fn num_output_tokens(&self) -> usize {
        self.output_token_ids.len()
    }

    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    pub fn get_finished_reason(&self) -> Option<FinishReason> {
        self.status.get_finished_reason()
    }

    pub fn get_num_encoder_tokens(&self, input_id: usize) -> usize {
        assert!(input_id < self.mm_features.len());
        self.mm_features[input_id].mm_position.length
    }

    pub fn record_event(&mut self, event_type: EngineCoreEventType, timestamp: Option<f64>) {
        self.events
            .push(EngineCoreEvent::new(event_type, timestamp));
    }

    pub fn take_events(&mut self) -> Option<Vec<EngineCoreEvent>> {
        if self.events.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.events))
        }
    }

    /// Create a simple request for testing purposes
    /// This mimics the old simple constructor used in tests
    #[cfg(test)]
    pub fn new_simple(request_id: String, num_tokens: usize) -> Self {
        // Create prompt tokens to match the number of tokens
        let prompt_token_ids = vec![1; num_tokens];

        Self::new(
            request_id,
            prompt_token_ids,
            Some(SamplingParams::default()),
            None,
            None,
            0,
            None,
            None,
            None,
            false, // use_structured_output
            0,
        )
    }
}
