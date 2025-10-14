use crate::core::types::request::Request;
use std::collections::{HashMap, HashSet, VecDeque};

/// Manages the encoder cache for multimodal and encoder-decoder models.
///
/// This cache stores encoder outputs (like image features, audio features, etc.)
/// to avoid recomputing them when they're reused across requests.
pub struct EncoderCacheManager {
    /// Total cache capacity in encoder tokens
    cache_size: usize,

    /// Current available cache capacity in encoder tokens
    num_free_slots: usize,

    /// Capacity that can be immediately reclaimed by evicting entries with zero references
    num_freeable_slots: usize,

    /// Map from mm_hash to a set of request IDs that currently reference the cached entry
    /// If the set is empty, the entry exists but is not referenced and is eligible for reclamation
    cached: HashMap<String, HashSet<String>>,

    /// List of (mm_hash, num_tokens) representing entries whose no current running request
    /// is needed and that can be freed to make space when needed (LRU order)
    freeable: VecDeque<(String, usize)>,

    /// Map from mm_hash to number of tokens (for quick lookup)
    token_counts: HashMap<String, usize>,

    /// List of freed multimodal hashes in this scheduling step
    freed_mm_hashes: Vec<String>,
}

impl EncoderCacheManager {
    /// Create a new encoder cache manager with the specified cache size
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            num_free_slots: cache_size,
            num_freeable_slots: cache_size,
            cached: HashMap::new(),
            freeable: VecDeque::new(),
            token_counts: HashMap::new(),
            freed_mm_hashes: Vec::new(),
        }
    }

    /// Check if an encoder input is already cached and update its reference count
    pub fn check_and_update_cache(&mut self, request: &Request, input_id: usize) -> bool {
        if input_id >= request.mm_features.len() {
            return false;
        }

        // Get the mm_hash from the multimodal feature identifier
        // For now using a simple hash based on request_id and input_id
        // In real implementation, this would be request.mm_features[input_id].identifier
        let mm_hash = format!("mm_{}_{}", request.request_id, input_id);

        // Not cached at all
        if !self.cached.contains_key(&mm_hash) {
            return false;
        }

        // Cached but currently not referenced by any request
        if self
            .cached
            .get(&mm_hash)
            .map(|s| s.is_empty())
            .unwrap_or(false)
        {
            // Remove from freeable list
            if let Some(pos) = self.freeable.iter().position(|(h, _)| h == &mm_hash) {
                let (_, num_tokens) = self.freeable.remove(pos).unwrap();
                self.num_freeable_slots -= num_tokens;
            }
        }

        self.cached
            .entry(mm_hash)
            .or_insert_with(HashSet::new)
            .insert(request.request_id.clone());
        true
    }

    /// Check if we can allocate space for an encoder input
    pub fn can_allocate(
        &mut self,
        request: &Request,
        input_id: usize,
        encoder_budget: usize,
        num_tokens_to_schedule: usize,
    ) -> bool {
        // Get the number of tokens needed for this encoder input
        let num_tokens = if input_id < request.mm_features.len() {
            request.mm_features[input_id].mm_position.length
        } else {
            return false;
        };

        // Check encoder compute budget
        if num_tokens > encoder_budget {
            return false;
        }

        let total_needed = num_tokens + num_tokens_to_schedule;

        // Enough free slots
        if total_needed <= self.num_free_slots {
            return true;
        }

        // Not enough reclaimable slots
        if total_needed > self.num_freeable_slots {
            return false;
        }

        // Not enough free slots but enough reclaimable slots - perform eviction
        while total_needed > self.num_free_slots {
            if let Some((mm_hash, num_free_tokens)) = self.freeable.pop_front() {
                self.cached.remove(&mm_hash);
                self.token_counts.remove(&mm_hash);
                self.freed_mm_hashes.push(mm_hash);
                self.num_free_slots += num_free_tokens;
            } else {
                return false;
            }
        }

        true
    }

    /// Allocate cache space for an encoder input
    pub fn allocate(&mut self, request: &Request, input_id: usize) {
        // Get the number of tokens for this encoder input
        let num_tokens = if input_id < request.mm_features.len() {
            request.mm_features[input_id].mm_position.length
        } else {
            return;
        };

        let mm_hash = format!("mm_{}_{}", request.request_id, input_id);
        let request_id = request.request_id.clone();

        // Add to cache if not already present
        if !self.cached.contains_key(&mm_hash) {
            self.cached.insert(mm_hash.clone(), HashSet::new());
            self.token_counts.insert(mm_hash.clone(), num_tokens);
        }

        // Encoder cache should always have enough space for encoder inputs
        // that are scheduled since eviction takes place at can_allocate()
        assert!(self.num_free_slots >= num_tokens);
        assert!(self.num_freeable_slots >= num_tokens);

        self.cached.get_mut(&mm_hash).unwrap().insert(request_id);
        self.num_free_slots -= num_tokens;
        self.num_freeable_slots -= num_tokens;
    }

    /// Get all cached multimodal input IDs for a request
    pub fn get_cached_input_ids(&self, request: &Request) -> HashSet<usize> {
        let mut cached_ids = HashSet::new();
        for input_id in 0..request.mm_features.len() {
            let mm_hash = format!("mm_{}_{}", request.request_id, input_id);
            if self.cached.contains_key(&mm_hash) {
                cached_ids.insert(input_id);
            }
        }
        cached_ids
    }

    /// Free a specific encoder input for a request
    pub fn free_encoder_input(&mut self, request: &Request, input_id: usize) {
        if input_id >= request.mm_features.len() {
            return;
        }

        let mm_hash = format!("mm_{}_{}", request.request_id, input_id);
        let req_id = &request.request_id;

        // The mm_hash not in cache or the req_id set is empty
        if !self
            .cached
            .get(&mm_hash)
            .map(|s| !s.is_empty())
            .unwrap_or(false)
        {
            return;
        }

        if let Some(ref_set) = self.cached.get_mut(&mm_hash) {
            ref_set.remove(req_id);

            if ref_set.is_empty() {
                // Move to freeable list
                if let Some(&num_tokens) = self.token_counts.get(&mm_hash) {
                    self.freeable.push_back((mm_hash, num_tokens));
                    self.num_freeable_slots += num_tokens;
                }
            }
        }
    }

    /// Free all encoder inputs for a request
    pub fn free(&mut self, request: &Request) {
        let cached_ids = self.get_cached_input_ids(request);
        for input_id in cached_ids {
            self.free_encoder_input(request, input_id);
        }
    }

    /// Get and clear the list of freed multimodal hashes
    pub fn get_freed_mm_hashes(&mut self) -> Vec<String> {
        let hashes = self.freed_mm_hashes.clone();
        self.freed_mm_hashes.clear();
        hashes
    }

    /// Get current cache usage
    pub fn usage(&self) -> f32 {
        if self.cache_size == 0 {
            0.0
        } else {
            (self.cache_size - self.num_free_slots) as f32 / self.cache_size as f32
        }
    }
}

/// Calculate encoder budget based on configuration
pub fn compute_encoder_budget(
    max_num_encoder_input_tokens: usize,
    is_encoder_decoder: bool,
) -> (usize, usize) {
    if is_encoder_decoder || max_num_encoder_input_tokens > 0 {
        // For encoder-decoder models or multimodal models,
        // use the configured budget for both compute and cache
        (max_num_encoder_input_tokens, max_num_encoder_input_tokens)
    } else {
        // No encoder budget needed for text-only models
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_cache_manager_basic() {
        let mut manager = EncoderCacheManager::new(1000);
        assert_eq!(manager.usage(), 0.0);

        // Create a mock request with multimodal features
        let mm_features = vec![crate::core::MultiModalFeatureSpec {
            identifier: "mm_req1_0".to_string(),
            mm_position: crate::core::PlaceholderRange {
                offset: 0,
                length: 100,
            },
            modality: "image".to_string(),
        }];

        let request = Request::new(
            "req1".to_string(),
            vec![1, 2, 3],
            Some(crate::core::SamplingParams::default()), // sampling_params
            None,                                         // pooling_params
            None,                                         // eos_token_id
            0,                                            // client_index
            None,                                         // arrival_time
            Some(mm_features),                            // mm_features
            None,                                         // lora_request
            false,                                        // use_structured_output
            0,                                            // priority
        );

        // Allocate encoder input
        assert!(manager.can_allocate(&request, 0, 200, 0));
        manager.allocate(&request, 0);
        assert_eq!(manager.num_free_slots, 900);

        // Check if cached
        assert!(manager.check_and_update_cache(&request, 0));

        // Free the input
        manager.free(&request);
        assert_eq!(manager.num_free_slots, 900); // Still allocated but freeable
        assert_eq!(manager.num_freeable_slots, 1000); // Now freeable

        // Should be in freeable list
        assert_eq!(manager.freeable.len(), 1);
    }

    #[test]
    fn test_compute_encoder_budget() {
        // Test encoder-decoder model
        let (compute, cache) = compute_encoder_budget(512, true);
        assert_eq!(compute, 512);
        assert_eq!(cache, 512);

        // Test text-only model
        let (compute, cache) = compute_encoder_budget(0, false);
        assert_eq!(compute, 0);
        assert_eq!(cache, 0);

        // Test multimodal model
        let (compute, cache) = compute_encoder_budget(256, false);
        assert_eq!(compute, 256);
        assert_eq!(cache, 256);
    }
}
