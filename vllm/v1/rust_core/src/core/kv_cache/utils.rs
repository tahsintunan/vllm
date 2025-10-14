use super::free_block_queue::BlockHash;
use crate::core::Request;
use lazy_static::lazy_static;
use sha2::{Digest, Sha256};
use std::sync::Mutex;

// Global hash seed for the first block of any prefix block sequence
lazy_static! {
    static ref NONE_HASH: Mutex<BlockHash> = {
        // Use PYTHONHASHSEED if set, otherwise use a default value
        let seed = std::env::var("PYTHONHASHSEED").unwrap_or_else(|_| "42".to_string());
        let hash = compute_sha256_hash(&seed.as_bytes());
        Mutex::new(hash)
    };
}

/// Compute SHA256 hash of the given data
pub fn compute_sha256_hash(data: &[u8]) -> BlockHash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Hash a block of tokens for prefix caching
///
/// Computes a hash value corresponding to the contents of a block and
/// the contents of the preceding block(s). The hash value is used for
/// prefix caching.
///
/// Args:
///     parent_block_hash: The hash of the parent block. None if this is the first block.
///     curr_block_token_ids: A list of token ids in the current block. The block is assumed to be full.
///     extra_keys: Extra keys for the block (for MM and LoRA requests).
///
/// Returns:
///     The hash value of the block.
pub fn hash_block_tokens(
    parent_block_hash: Option<&BlockHash>,
    curr_block_token_ids: &[i32],
    extra_keys: Option<&[String]>,
) -> BlockHash {
    let mut hasher = Sha256::new();

    // Use parent hash or NONE_HASH
    match parent_block_hash {
        Some(hash) => hasher.update(hash),
        None => hasher.update(&*NONE_HASH.lock().unwrap()),
    }

    // Add token IDs
    for token_id in curr_block_token_ids {
        hasher.update(token_id.to_le_bytes());
    }

    // Add extra keys if present
    if let Some(keys) = extra_keys {
        for key in keys {
            hasher.update(key.as_bytes());
        }
    }

    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Compute block hashes for a request
///
/// Returns a list of un-computed block hashes for a request.
/// Only computes hashes for full blocks.
pub fn compute_request_block_hashes(request: &Request, block_size: usize) -> Vec<BlockHash> {
    let start_token_idx = request.block_hashes.len() * block_size;
    let num_tokens = request.all_token_ids.len();

    let mut prev_block_hash: Option<BlockHash> = if !request.block_hashes.is_empty() {
        Some(*request.block_hashes.last().unwrap())
    } else {
        None
    };

    let mut new_block_hashes = Vec::new();
    let mut current_idx = start_token_idx;

    while current_idx + block_size <= num_tokens {
        let end_idx = current_idx + block_size;

        // Get tokens for this block
        let block_tokens = &request.all_token_ids[current_idx..end_idx];

        // Generate extra keys for MM and LoRA requests if needed
        let extra_keys = generate_block_hash_extra_keys(request, current_idx, end_idx);

        // Compute the hash of the current block
        let block_hash = hash_block_tokens(
            prev_block_hash.as_ref(),
            block_tokens,
            extra_keys.as_deref(),
        );

        new_block_hashes.push(block_hash);
        current_idx += block_size;
        prev_block_hash = Some(block_hash);
    }

    new_block_hashes
}

/// Generate extra keys for block hash computation
///
/// This handles multimodal and LoRA request-specific keys that need to be
/// included in the block hash for correctness.
fn generate_block_hash_extra_keys(
    request: &Request,
    start_idx: usize,
    _end_idx: usize, // TODO: why is end_idx unused? (needed for future use cases?)
) -> Option<Vec<String>> {
    let mut extra_keys = Vec::new();

    // Check if there are multimodal features in this block range
    for (i, mm_feature) in request.mm_features.iter().enumerate() {
        // For simplicity, we'll include the MM feature type if it affects this block
        // In a real implementation, this would check the actual position of MM inputs
        if !request.mm_features.is_empty() && start_idx == 0 {
            extra_keys.push(format!(
                "mm_{}_{}_{}",
                i, mm_feature.feature_type, mm_feature.num_tokens
            ));
        }
    }

    // Check for LoRA request
    if let Some(lora_request) = &request.lora_request {
        extra_keys.push(format!("lora_{}", lora_request.lora_int_id));
    }

    if extra_keys.is_empty() {
        None
    } else {
        Some(extra_keys)
    }
}

/// Prefix caching metrics for tracking hit rates
pub struct PrefixCachingMetrics {
    /// Number of cache hits
    hits: usize,
    /// Number of cache misses
    misses: usize,
    /// Total number of cached blocks
    cached_blocks: usize,
    /// Recent request hit rates
    recent_hit_rates: Vec<f32>,
    /// Maximum number of recent requests to track
    max_recent_requests: usize,
}

impl PrefixCachingMetrics {
    pub fn new(max_recent_requests: usize) -> Self {
        Self {
            hits: 0,
            misses: 0,
            cached_blocks: 0,
            recent_hit_rates: Vec::new(),
            max_recent_requests,
        }
    }

    pub fn record_hit(&mut self, num_blocks: usize) {
        self.hits += 1;
        self.cached_blocks += num_blocks;
        self.update_recent_hit_rate(true);
    }

    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.update_recent_hit_rate(false);
    }

    fn update_recent_hit_rate(&mut self, is_hit: bool) {
        let rate = if is_hit { 1.0 } else { 0.0 };
        self.recent_hit_rates.push(rate);

        if self.recent_hit_rates.len() > self.max_recent_requests {
            self.recent_hit_rates.remove(0);
        }
    }

    pub fn get_hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    pub fn get_recent_hit_rate(&self) -> f32 {
        if self.recent_hit_rates.is_empty() {
            0.0
        } else {
            let sum: f32 = self.recent_hit_rates.iter().sum();
            sum / self.recent_hit_rates.len() as f32
        }
    }

    pub fn get_cached_blocks(&self) -> usize {
        self.cached_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MultiModalFeatureSpec, Request, SamplingParams};

    #[test]
    fn test_hash_block_tokens() {
        let tokens = vec![1, 2, 3, 4, 5];

        // First block with no parent
        let hash1 = hash_block_tokens(None, &tokens, None);
        assert!(!hash1.is_empty());

        // Second block with parent
        let hash2 = hash_block_tokens(Some(&hash1), &tokens, None);
        assert!(!hash2.is_empty());
        assert_ne!(hash1, hash2);

        // Same tokens but with extra keys
        let extra_keys = vec!["key1".to_string(), "key2".to_string()];
        let hash3 = hash_block_tokens(Some(&hash1), &tokens, Some(&extra_keys));
        assert_ne!(hash2, hash3);
    }

    #[test]
    fn test_compute_request_block_hashes() {
        let request = Request::new(
            "test_req".to_string(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Some(SamplingParams::default()),
            None,
            None,
            0,
            None,
            None,
            None,
            false, // use_structured_output
            0,
        );

        // Compute hashes for block size 4
        let hashes = compute_request_block_hashes(&request, 4);
        assert_eq!(hashes.len(), 2); // 2 full blocks (8 tokens), 2 remaining

        // Hashes should be different
        assert_ne!(hashes[0], hashes[1]);
    }

    #[test]
    fn test_prefix_caching_metrics() {
        let mut metrics = PrefixCachingMetrics::new(10);

        // Record some hits and misses
        metrics.record_hit(5);
        metrics.record_hit(3);
        metrics.record_miss();

        assert_eq!(metrics.get_hit_rate(), 2.0 / 3.0);
        assert_eq!(metrics.get_cached_blocks(), 8);

        // Recent hit rate should reflect recent activity
        assert!((metrics.get_recent_hit_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_deterministic_hashing() {
        // Test that hashing is deterministic
        let tokens = vec![1, 2, 3, 4, 5];
        let hash1 = hash_block_tokens(None, &tokens, None);
        let hash2 = hash_block_tokens(None, &tokens, None);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_multimodal_extra_keys() {
        let request = Request::new(
            "test_mm".to_string(),
            vec![1, 2, 3, 4],
            Some(SamplingParams::default()),
            None,
            None,
            0,
            None,
            Some(vec![MultiModalFeatureSpec {
                feature_type: "image".to_string(),
                num_tokens: 100,
                identifier: "test_image_0".to_string(),
                mm_position: crate::core::PlaceholderRange {
                    offset: 0,
                    length: 100,
                },
            }]),
            None,
            false, // use_structured_output
            0,
        );

        // Generate extra keys for first block
        let extra_keys = generate_block_hash_extra_keys(&request, 0, 4);
        assert!(extra_keys.is_some());
        assert!(!extra_keys.unwrap().is_empty());
    }
}
