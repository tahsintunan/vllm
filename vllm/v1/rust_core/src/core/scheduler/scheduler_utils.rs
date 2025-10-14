use crate::core::types::request::{Request, RequestStatus, StopReason};

/// Check if a request should be stopped based on various conditions.
///
/// Checks for:
/// - Maximum model length exceeded
/// - Maximum output tokens reached
/// - Pooling request completion
/// - EOS token encountered
/// - Stop token encountered
///
/// Returns true if the request should stop, false otherwise.
/// Also updates the request status accordingly.
pub fn check_stop(
    request: &mut Request,
    max_model_len: usize,
    pooler_output: Option<&Vec<f32>>,
) -> bool {
    // Check if we've hit model or output token limits
    if request.num_tokens() >= max_model_len
        || request.num_output_tokens() >= request.max_tokens as usize
    {
        request.status = RequestStatus::FinishedLengthCapped;
        return true;
    }

    // Check pooling params
    if let Some(_pooling_params) = &request.pooling_params {
        if pooler_output.is_some() {
            request.status = RequestStatus::FinishedStopped;
            return true;
        }
        return false;
    }

    // Check sampling params for stop conditions
    if let Some(sampling_params) = &request.sampling_params {
        if !request.output_token_ids.is_empty() {
            let last_token_id = *request.output_token_ids.last().unwrap();

            // Check for EOS token
            if let Some(eos_token_id) = request.eos_token_id {
                if !sampling_params.ignore_eos && last_token_id == eos_token_id {
                    request.status = RequestStatus::FinishedStopped;
                    return true;
                }
            }

            // Check for stop tokens
            if let Some(stop_token_ids) = &sampling_params.stop_token_ids {
                if stop_token_ids.contains(&last_token_id) {
                    request.status = RequestStatus::FinishedStopped;
                    request.stop_reason = Some(StopReason::TokenId(last_token_id));
                    return true;
                }
            }
        }
    }

    false
}

/// Helper structure for token budget management
pub struct TokenBudget {
    pub max_tokens: usize,
    pub remaining: usize,
}

impl TokenBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            remaining: max_tokens,
        }
    }

    /// Try to allocate tokens from the budget
    /// Returns the actual number of tokens allocated (may be less than requested)
    pub fn allocate(&mut self, requested: usize) -> usize {
        let allocated = requested.min(self.remaining);
        self.remaining -= allocated;
        allocated
    }

    /// Check if budget has any tokens remaining
    pub fn has_tokens(&self) -> bool {
        self.remaining > 0
    }

    /// Get the number of tokens used
    pub fn used(&self) -> usize {
        self.max_tokens - self.remaining
    }
}

/// Helper for managing encoder compute budget
pub struct EncoderBudget {
    pub max_tokens: usize,
    pub remaining: usize,
}

impl EncoderBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            remaining: max_tokens,
        }
    }

    /// Try to allocate encoder tokens from the budget
    pub fn allocate(&mut self, requested: usize) -> Option<usize> {
        if requested <= self.remaining {
            self.remaining -= requested;
            Some(requested)
        } else {
            None
        }
    }

    /// Check if budget can accommodate the requested tokens
    pub fn can_allocate(&self, requested: usize) -> bool {
        requested <= self.remaining
    }
}

/// Calculate the number of tokens to schedule for chunked prefill
pub fn calculate_chunked_prefill_tokens(
    num_new_tokens: usize,
    long_prefill_threshold: usize,
    token_budget: usize,
) -> usize {
    let mut tokens = num_new_tokens;

    // Apply long prefill threshold if configured
    if long_prefill_threshold > 0 && tokens > long_prefill_threshold {
        tokens = long_prefill_threshold;
    }

    // Respect token budget
    tokens.min(token_budget)
}

/// Check if a request qualifies for prefix caching
pub fn should_use_prefix_cache(
    request: &Request,
    enable_prefix_caching: bool,
) -> bool {
    if !enable_prefix_caching {
        return false;
    }

    // Check if request has sufficient prompt tokens for caching to be beneficial
    // This threshold can be configured based on performance characteristics
    const MIN_PROMPT_TOKENS_FOR_CACHE: usize = 32;

    request.num_prompt_tokens >= MIN_PROMPT_TOKENS_FOR_CACHE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_budget() {
        let mut budget = TokenBudget::new(100);
        assert_eq!(budget.allocate(50), 50);
        assert_eq!(budget.remaining, 50);
        assert_eq!(budget.allocate(60), 50); // Only 50 remaining
        assert_eq!(budget.remaining, 0);
        assert!(!budget.has_tokens());
        assert_eq!(budget.used(), 100);
    }

    #[test]
    fn test_encoder_budget() {
        let mut budget = EncoderBudget::new(100);
        assert!(budget.can_allocate(50));
        assert_eq!(budget.allocate(50), Some(50));
        assert_eq!(budget.remaining, 50);
        assert!(!budget.can_allocate(60));
        assert_eq!(budget.allocate(60), None);
        assert_eq!(budget.remaining, 50); // Unchanged on failed allocation
    }

    #[test]
    fn test_calculate_chunked_prefill_tokens() {
        // Test without threshold
        assert_eq!(calculate_chunked_prefill_tokens(100, 0, 200), 100);

        // Test with threshold
        assert_eq!(calculate_chunked_prefill_tokens(100, 50, 200), 50);

        // Test with budget constraint
        assert_eq!(calculate_chunked_prefill_tokens(100, 150, 80), 80);
    }
}