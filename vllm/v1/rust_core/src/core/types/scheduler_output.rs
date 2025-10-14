use std::collections::{HashMap, HashSet};

use super::request::{LoraRequest, MultiModalFeatureSpec, PoolingParams, Request, SamplingParams};

/// Data for a new request being scheduled for the first time
#[derive(Debug, Clone)]
pub struct NewRequestData {
    /// Request ID
    pub req_id: String,
    /// Prompt token IDs
    pub prompt_token_ids: Vec<i32>,
    /// Multimodal features
    pub mm_features: Vec<MultiModalFeatureSpec>,
    /// Sampling parameters
    pub sampling_params: Option<SamplingParams>,
    /// Pooling parameters
    pub pooling_params: Option<PoolingParams>,
    /// Block IDs allocated for this request
    /// tuple of lists, each list contains block IDs for a layer
    pub block_ids: Vec<Vec<u32>>,
    /// Number of tokens already computed
    pub num_computed_tokens: usize,
    /// LoRA request if applicable
    pub lora_request: Option<LoraRequest>,
}

impl NewRequestData {
    /// Create from a Request and allocated block IDs
    pub fn from_request(request: &Request, block_ids: Vec<Vec<u32>>) -> Self {
        Self {
            req_id: request.request_id.clone(),
            prompt_token_ids: request.prompt_token_ids.clone(),
            mm_features: Vec::new(), // Will be populated based on request.mm_features
            sampling_params: request.sampling_params.clone(),
            pooling_params: None, // Will be populated based on request.pooling_params
            block_ids,
            num_computed_tokens: request.num_computed_tokens,
            lora_request: None, // Will be populated based on request.lora_request
        }
    }

    /// Anonymous representation with prompt data obfuscated
    pub fn anon_repr(&self) -> String {
        format!(
            "NewRequestData(req_id={}, prompt_token_ids_len={}, mm_features={:?}, \
             sampling_params={:?}, block_ids={:?}, num_computed_tokens={}, \
             lora_request={:?})",
            self.req_id,
            self.prompt_token_ids.len(),
            self.mm_features,
            self.sampling_params,
            self.block_ids,
            self.num_computed_tokens,
            self.lora_request
        )
    }
}

/// Data for cached requests that have been scheduled before
#[derive(Debug, Clone)]
pub struct CachedRequestData {
    /// Request IDs
    pub req_ids: Vec<String>,
    /// Whether each request was resumed from preemption
    /// If false, new_block_ids will be appended to the request's block IDs
    /// If true, new_block_ids will replace the request's block IDs
    pub resumed_from_preemption: Vec<bool>,
    /// New token IDs (only used for pipeline parallelism)
    /// When PP is not used, this will be empty
    pub new_token_ids: Vec<Vec<i32>>,
    /// New block IDs for each request
    /// None if no new blocks allocated
    pub new_block_ids: Vec<Option<Vec<Vec<u32>>>>,
    /// Number of computed tokens for each request
    pub num_computed_tokens: Vec<usize>,
}

impl CachedRequestData {
    /// Get the number of requests
    pub fn num_reqs(&self) -> usize {
        self.req_ids.len()
    }

    /// Create an empty CachedRequestData
    pub fn make_empty() -> Self {
        Self {
            req_ids: Vec::new(),
            resumed_from_preemption: Vec::new(),
            new_token_ids: Vec::new(),
            new_block_ids: Vec::new(),
            num_computed_tokens: Vec::new(),
        }
    }
}

/// Abstract trait for KV connector metadata
/// This is a placeholder - actual implementation will depend on transport layer
pub trait KVConnectorMetadata: std::fmt::Debug {
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Main scheduler output containing all scheduling decisions
#[derive(Debug)]
pub struct SchedulerOutput {
    /// List of requests scheduled for the first time
    /// We cache the request's data in each worker process
    pub scheduled_new_reqs: Vec<NewRequestData>,

    /// List of requests that have been scheduled before
    /// Only send the diff to minimize communication cost
    pub scheduled_cached_reqs: CachedRequestData,

    /// req_id -> num_scheduled_tokens
    /// Number of tokens scheduled for each request
    pub num_scheduled_tokens: HashMap<String, usize>,

    /// Total number of tokens scheduled for all requests
    /// Equal to sum of num_scheduled_tokens.values()
    pub total_num_scheduled_tokens: usize,

    /// req_id -> spec_token_ids
    /// Speculative decoding tokens for requests that have them
    pub scheduled_spec_decode_tokens: HashMap<String, Vec<i32>>,

    /// req_id -> encoder input indices that need processing
    /// E.g., [0, 1] means the vision encoder needs to process
    /// the request's 0-th and 1-th images in the current step
    pub scheduled_encoder_inputs: HashMap<String, Vec<usize>>,

    /// Number of common prefix blocks for all requests in each KV cache group
    /// Used for cascade attention optimization
    pub num_common_prefix_blocks: Vec<usize>,

    /// Request IDs that finished between previous and current steps
    /// Used to notify workers to free cached states
    pub finished_req_ids: HashSet<String>,

    /// List of mm_hash strings for encoder outputs to be freed from cache
    pub free_encoder_mm_hashes: Vec<String>,

    /// Dict of request IDs to their index within the batch
    /// Used for filling the next token bitmask
    pub structured_output_request_ids: HashMap<String, usize>,

    /// Bitmask for the whole batch (for structured output generation)
    /// In Python this is Optional[npt.ANDArray[np.int32]]
    /// We use Vec<Vec<i32>> to represent 2D array
    pub grammar_bitmask: Option<Vec<Vec<i32>>>,

    /// KV Cache Connector metadata (optional)
    pub kv_connector_metadata: Option<Box<dyn KVConnectorMetadata>>,
}

impl SchedulerOutput {
    /// Create a new empty SchedulerOutput
    pub fn new() -> Self {
        Self {
            scheduled_new_reqs: Vec::new(),
            scheduled_cached_reqs: CachedRequestData::make_empty(),
            num_scheduled_tokens: HashMap::new(),
            total_num_scheduled_tokens: 0,
            scheduled_spec_decode_tokens: HashMap::new(),
            scheduled_encoder_inputs: HashMap::new(),
            num_common_prefix_blocks: Vec::new(),
            finished_req_ids: HashSet::new(),
            free_encoder_mm_hashes: Vec::new(),
            structured_output_request_ids: HashMap::new(),
            grammar_bitmask: None,
            kv_connector_metadata: None,
        }
    }

    /// Add a new request to the scheduler output
    pub fn add_new_request(&mut self, request_data: NewRequestData, num_tokens: usize) {
        self.num_scheduled_tokens
            .insert(request_data.req_id.clone(), num_tokens);
        self.total_num_scheduled_tokens += num_tokens;
        self.scheduled_new_reqs.push(request_data);
    }

    /// Add a cached request to the scheduler output
    pub fn add_cached_request(
        &mut self,
        req_id: String,
        resumed_from_preemption: bool,
        new_token_ids: Vec<i32>,
        new_block_ids: Option<Vec<Vec<u32>>>,
        num_computed_tokens: usize,
        num_scheduled_tokens: usize,
    ) {
        self.scheduled_cached_reqs.req_ids.push(req_id.clone());
        self.scheduled_cached_reqs
            .resumed_from_preemption
            .push(resumed_from_preemption);
        self.scheduled_cached_reqs.new_token_ids.push(new_token_ids);
        self.scheduled_cached_reqs.new_block_ids.push(new_block_ids);
        self.scheduled_cached_reqs
            .num_computed_tokens
            .push(num_computed_tokens);

        self.num_scheduled_tokens
            .insert(req_id, num_scheduled_tokens);
        self.total_num_scheduled_tokens += num_scheduled_tokens;
    }

    /// Mark a request as finished
    pub fn mark_finished(&mut self, req_id: String) {
        self.finished_req_ids.insert(req_id);
    }

    /// Add speculative decode tokens for a request
    pub fn add_spec_decode_tokens(&mut self, req_id: String, tokens: Vec<i32>) {
        self.scheduled_spec_decode_tokens.insert(req_id, tokens);
    }

    /// Add encoder inputs for a request
    pub fn add_encoder_inputs(&mut self, req_id: String, indices: Vec<usize>) {
        self.scheduled_encoder_inputs.insert(req_id, indices);
    }

    /// Check if the output is empty (no requests scheduled)
    pub fn is_empty(&self) -> bool {
        self.scheduled_new_reqs.is_empty() && self.scheduled_cached_reqs.num_reqs() == 0
    }

    /// Get total number of scheduled requests
    pub fn num_scheduled_reqs(&self) -> usize {
        self.scheduled_new_reqs.len() + self.scheduled_cached_reqs.num_reqs()
    }
}

impl Default for SchedulerOutput {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request(id: &str) -> Request {
        Request {
            request_id: id.to_string(),
            prompt_token_ids: vec![1, 2, 3],
            sampling_params: Some(SamplingParams {
                max_tokens: Some(100),
                ..Default::default()
            }),
            pooling_params: None,
            eos_token_id: Some(2),
            client_index: 0,
            arrival_time: 1000.0,
            mm_features: Vec::new(),
            lora_request: None,
            use_structured_output: false,
            priority: 0,

            // State fields
            status: crate::core::types::request::RequestStatus::Waiting,
            events: Vec::new(),
            stop_reason: None,

            // Token management
            max_tokens: 100,
            num_prompt_tokens: 3,
            output_token_ids: Vec::new(),
            all_token_ids: vec![1, 2, 3],
            num_output_placeholders: 0,
            spec_token_ids: Vec::new(),
            num_computed_tokens: 0,

            // Multi-modal related
            has_encoder_inputs: false,

            // Caching related
            num_cached_tokens: 0,
            block_hashes: Vec::new(),

            // Prompt logprobs
            prompt_logprobs_enabled: false,
        }
    }

    #[test]
    fn test_new_request_data_creation() {
        let request = create_test_request("test-123");
        let block_ids = vec![vec![1u32, 2], vec![3, 4]];

        let new_req_data = NewRequestData::from_request(&request, block_ids.clone());

        assert_eq!(new_req_data.req_id, "test-123");
        assert_eq!(new_req_data.prompt_token_ids, vec![1, 2, 3]);
        assert_eq!(new_req_data.block_ids, block_ids);
        assert_eq!(new_req_data.num_computed_tokens, 0);
        assert!(new_req_data.sampling_params.is_some());
    }

    #[test]
    fn test_new_request_data_anon_repr() {
        let request = create_test_request("secret-req");
        let block_ids = vec![vec![10u32, 20]];

        let new_req_data = NewRequestData::from_request(&request, block_ids);
        let anon_repr = new_req_data.anon_repr();

        assert!(anon_repr.contains("secret-req"));
        assert!(anon_repr.contains("prompt_token_ids_len=3"));
        assert!(!anon_repr.contains("[1, 2, 3]")); // Should not contain actual tokens
    }

    #[test]
    fn test_cached_request_data_empty() {
        let cached_data = CachedRequestData::make_empty();

        assert_eq!(cached_data.num_reqs(), 0);
        assert!(cached_data.req_ids.is_empty());
        assert!(cached_data.resumed_from_preemption.is_empty());
        assert!(cached_data.new_token_ids.is_empty());
        assert!(cached_data.new_block_ids.is_empty());
        assert!(cached_data.num_computed_tokens.is_empty());
    }

    #[test]
    fn test_cached_request_data_with_requests() {
        let mut cached_data = CachedRequestData::make_empty();

        cached_data.req_ids.push("req-1".to_string());
        cached_data.req_ids.push("req-2".to_string());
        cached_data.resumed_from_preemption.push(false);
        cached_data.resumed_from_preemption.push(true);
        cached_data.new_token_ids.push(vec![100]);
        cached_data.new_token_ids.push(vec![200, 201]);
        cached_data.new_block_ids.push(Some(vec![vec![1u32, 2]]));
        cached_data.new_block_ids.push(None);
        cached_data.num_computed_tokens.push(10);
        cached_data.num_computed_tokens.push(20);

        assert_eq!(cached_data.num_reqs(), 2);
        assert_eq!(cached_data.req_ids[0], "req-1");
        assert_eq!(cached_data.req_ids[1], "req-2");
        assert!(!cached_data.resumed_from_preemption[0]);
        assert!(cached_data.resumed_from_preemption[1]);
    }

    #[test]
    fn test_scheduler_output_new_empty() {
        let output = SchedulerOutput::new();

        assert!(output.is_empty());
        assert_eq!(output.num_scheduled_reqs(), 0);
        assert_eq!(output.total_num_scheduled_tokens, 0);
        assert!(output.scheduled_new_reqs.is_empty());
        assert_eq!(output.scheduled_cached_reqs.num_reqs(), 0);
        assert!(output.finished_req_ids.is_empty());
    }

    #[test]
    fn test_scheduler_output_add_new_request() {
        let mut output = SchedulerOutput::new();
        let request = create_test_request("new-req");
        let block_ids = vec![vec![5u32, 6]];
        let new_req_data = NewRequestData::from_request(&request, block_ids);

        output.add_new_request(new_req_data.clone(), 10);

        assert!(!output.is_empty());
        assert_eq!(output.num_scheduled_reqs(), 1);
        assert_eq!(output.total_num_scheduled_tokens, 10);
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "new-req");
        assert_eq!(*output.num_scheduled_tokens.get("new-req").unwrap(), 10);
    }

    #[test]
    fn test_scheduler_output_add_cached_request() {
        let mut output = SchedulerOutput::new();

        output.add_cached_request(
            "cached-req".to_string(),
            false,                     // not resumed from preemption
            vec![500, 501],            // new token ids
            Some(vec![vec![7u32, 8]]), // new block ids
            15,                        // num computed tokens
            5,                         // num scheduled tokens
        );

        assert!(!output.is_empty());
        assert_eq!(output.num_scheduled_reqs(), 1);
        assert_eq!(output.total_num_scheduled_tokens, 5);
        assert_eq!(output.scheduled_cached_reqs.num_reqs(), 1);
        assert_eq!(output.scheduled_cached_reqs.req_ids[0], "cached-req");
        assert!(!output.scheduled_cached_reqs.resumed_from_preemption[0]);
        assert_eq!(
            output.scheduled_cached_reqs.new_token_ids[0],
            vec![500, 501]
        );
        assert_eq!(output.scheduled_cached_reqs.num_computed_tokens[0], 15);
    }

    #[test]
    fn test_scheduler_output_multiple_requests() {
        let mut output = SchedulerOutput::new();

        // Add new requests
        let req1 = create_test_request("req-1");
        let req2 = create_test_request("req-2");
        output.add_new_request(NewRequestData::from_request(&req1, vec![vec![1u32]]), 10);
        output.add_new_request(NewRequestData::from_request(&req2, vec![vec![2u32]]), 20);

        // Add cached requests
        output.add_cached_request("req-3".to_string(), true, vec![], None, 5, 15);

        assert_eq!(output.num_scheduled_reqs(), 3);
        assert_eq!(output.total_num_scheduled_tokens, 10 + 20 + 15);
        assert_eq!(output.scheduled_new_reqs.len(), 2);
        assert_eq!(output.scheduled_cached_reqs.num_reqs(), 1);
    }

    #[test]
    fn test_scheduler_output_mark_finished() {
        let mut output = SchedulerOutput::new();

        output.mark_finished("req-1".to_string());
        output.mark_finished("req-2".to_string());

        assert_eq!(output.finished_req_ids.len(), 2);
        assert!(output.finished_req_ids.contains("req-1"));
        assert!(output.finished_req_ids.contains("req-2"));
    }

    #[test]
    fn test_scheduler_output_spec_decode_tokens() {
        let mut output = SchedulerOutput::new();

        output.add_spec_decode_tokens("req-1".to_string(), vec![100, 101, 102]);
        output.add_spec_decode_tokens("req-2".to_string(), vec![200]);

        assert_eq!(output.scheduled_spec_decode_tokens.len(), 2);
        assert_eq!(
            output.scheduled_spec_decode_tokens.get("req-1").unwrap(),
            &vec![100, 101, 102]
        );
        assert_eq!(
            output.scheduled_spec_decode_tokens.get("req-2").unwrap(),
            &vec![200]
        );
    }

    #[test]
    fn test_scheduler_output_encoder_inputs() {
        let mut output = SchedulerOutput::new();

        output.add_encoder_inputs("req-1".to_string(), vec![0, 1]);
        output.add_encoder_inputs("req-2".to_string(), vec![2, 3, 4]);

        assert_eq!(output.scheduled_encoder_inputs.len(), 2);
        assert_eq!(
            output.scheduled_encoder_inputs.get("req-1").unwrap(),
            &vec![0, 1]
        );
        assert_eq!(
            output.scheduled_encoder_inputs.get("req-2").unwrap(),
            &vec![2, 3, 4]
        );
    }

    #[test]
    fn test_scheduler_output_free_encoder_mm_hashes() {
        let mut output = SchedulerOutput::new();

        output.free_encoder_mm_hashes.push("hash1".to_string());
        output.free_encoder_mm_hashes.push("hash2".to_string());

        assert_eq!(output.free_encoder_mm_hashes.len(), 2);
        assert_eq!(output.free_encoder_mm_hashes[0], "hash1");
        assert_eq!(output.free_encoder_mm_hashes[1], "hash2");
    }

    #[test]
    fn test_scheduler_output_structured_output_request_ids() {
        let mut output = SchedulerOutput::new();

        output
            .structured_output_request_ids
            .insert("req-1".to_string(), 0);
        output
            .structured_output_request_ids
            .insert("req-2".to_string(), 1);
        output
            .structured_output_request_ids
            .insert("req-3".to_string(), 2);

        assert_eq!(output.structured_output_request_ids.len(), 3);
        assert_eq!(
            *output.structured_output_request_ids.get("req-1").unwrap(),
            0
        );
        assert_eq!(
            *output.structured_output_request_ids.get("req-2").unwrap(),
            1
        );
        assert_eq!(
            *output.structured_output_request_ids.get("req-3").unwrap(),
            2
        );
    }

    #[test]
    fn test_scheduler_output_grammar_bitmask() {
        let mut output = SchedulerOutput::new();

        // Test None case
        assert!(output.grammar_bitmask.is_none());

        // Test Some case with 2D array
        let bitmask = vec![vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0]];
        output.grammar_bitmask = Some(bitmask.clone());

        assert!(output.grammar_bitmask.is_some());
        assert_eq!(output.grammar_bitmask.unwrap(), bitmask);
    }

    #[test]
    fn test_scheduler_output_common_prefix_blocks() {
        let mut output = SchedulerOutput::new();

        output.num_common_prefix_blocks.push(5);
        output.num_common_prefix_blocks.push(3);
        output.num_common_prefix_blocks.push(7);

        assert_eq!(output.num_common_prefix_blocks.len(), 3);
        assert_eq!(output.num_common_prefix_blocks[0], 5);
        assert_eq!(output.num_common_prefix_blocks[1], 3);
        assert_eq!(output.num_common_prefix_blocks[2], 7);
    }

    #[test]
    fn test_scheduler_output_default() {
        let output = SchedulerOutput::default();

        assert!(output.is_empty());
        assert_eq!(output.num_scheduled_reqs(), 0);
        assert_eq!(output.total_num_scheduled_tokens, 0);
    }
}
