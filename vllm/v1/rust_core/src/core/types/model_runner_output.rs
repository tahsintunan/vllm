use std::collections::HashMap;

/// Output from the model runner after processing a batch of requests.
///
/// This contains the results of a single forward pass through the model,
/// including generated tokens, log probabilities, and other outputs.
pub struct ModelRunnerOutput {
    /// Generated token IDs for each request in the batch
    pub sampled_token_ids: Vec<Vec<i32>>,

    /// Log probabilities for the generated tokens (if requested)
    pub logprobs: Option<Vec<f32>>,

    /// Prompt log probabilities indexed by request ID
    pub prompt_logprobs_dict: HashMap<String, Vec<f32>>,

    /// Pooler output for embedding models
    pub pooler_output: Option<Vec<f32>>,

    /// Count of NaN values in logits per request (for debugging)
    pub num_nans_in_logits: Option<HashMap<String, usize>>,

    /// Mapping from request ID to index in the batch
    pub req_id_to_index: HashMap<String, usize>,

    /// KV cache transfer output (for distributed execution)
    pub kv_connector_output: Option<KvConnectorOutput>,
}

/// Output from the KV cache connector for distributed execution
pub struct KvConnectorOutput {
    // Placeholder for KV connector output data
    // This will be expanded when implementing distributed execution
}

/// Draft token IDs for speculative decoding
pub struct DraftTokenIds {
    /// Request IDs for the draft tokens
    pub req_ids: Vec<String>,

    /// Draft token IDs for each request
    pub draft_token_ids: Vec<Vec<i32>>,
}