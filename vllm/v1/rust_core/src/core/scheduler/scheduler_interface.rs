use std::collections::HashMap;

use crate::core::types::{
    engine_core_outputs::{EngineCoreOutputs, SchedulerStats},
    model_runner_output::{DraftTokenIds, ModelRunnerOutput},
    request::{Request, RequestStatus},
    scheduler_output::SchedulerOutput,
};

/// The main interface that all schedulers must implement.
///
/// This trait defines the core scheduling operations required for managing
/// request lifecycles, from addition to the queue through execution and completion.
pub trait SchedulerInterface {
    /// Schedule the requests to process in this scheduling step.
    ///
    /// The scheduling decision is made at the iteration level. Each scheduling
    /// step corresponds to a single forward pass of the model. Therefore, this
    /// method is called repeatedly by a busy loop in the engine.
    ///
    /// Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
    /// that specifies how many tokens to process for each request in this
    /// scheduling step. For example, num_tokens can be as large as the number
    /// of prompt tokens for new requests, or it can be 1 for the requests that
    /// are auto-regressively generating new tokens one by one. Otherwise, it
    /// can be somewhere in between in case of chunked prefills, prefix caching,
    /// speculative decoding, etc.
    ///
    /// Additionally, the scheduler also returns useful data about each request
    /// or the batch as a whole. The model runner will use this information in
    /// preparing inputs to the model.
    ///
    /// Returns:
    ///     A SchedulerOutput object containing information about the scheduled
    ///     requests.
    fn schedule(&mut self) -> SchedulerOutput;

    /// Update the scheduler state based on the model runner output.
    ///
    /// This method is called after the model runner has processed the scheduled
    /// requests. The model runner output includes generated token ids, draft
    /// token ids for next step, etc. The scheduler uses this information to
    /// update its states, checks the finished requests, and returns the output
    /// for each request.
    ///
    /// Returns:
    ///     A dict of client index to EngineCoreOutputs object containing the
    ///     outputs for each request originating from that client.
    fn update_from_output(
        &mut self,
        scheduler_output: &SchedulerOutput,
        model_runner_output: &ModelRunnerOutput,
    ) -> HashMap<i32, EngineCoreOutputs>;

    /// Update the draft token ids for the scheduled requests.
    fn update_draft_token_ids(&mut self, draft_token_ids: &DraftTokenIds);

    /// Add a new request to the scheduler's internal queue.
    ///
    /// Args:
    ///     request: The new request being added.
    fn add_request(&mut self, request: Request);

    /// Finish the requests in the scheduler's internal queue. If the request
    /// is not in the queue, this method will do nothing.
    ///
    /// This method is called in two cases:
    /// 1. When the request is aborted by the client.
    /// 2. When the frontend process detects a stop string of the request after
    ///    de-tokenizing its generated tokens.
    ///
    /// Args:
    ///     request_ids: A single or a list of request IDs.
    ///     finished_status: The finished status of the given requests.
    fn finish_requests(&mut self, request_ids: Vec<String>, finished_status: RequestStatus);

    /// Number of unfinished requests in the scheduler's internal queue.
    fn get_num_unfinished_requests(&self) -> usize;

    /// Returns true if there are unfinished requests in the scheduler's
    /// internal queue.
    fn has_unfinished_requests(&self) -> bool {
        self.get_num_unfinished_requests() > 0
    }

    /// Returns true if there are finished requests that need to be cleared.
    /// NOTE: This is different from `!self.has_unfinished_requests()`.
    ///
    /// The scheduler maintains an internal list of the requests finished in the
    /// previous step. This list is returned from the next call to schedule(),
    /// to be sent to the model runner in the next step to clear cached states
    /// for these finished requests.
    ///
    /// This method checks if this internal list of finished requests is
    /// non-empty. This information is useful for DP attention.
    fn has_finished_requests(&self) -> bool;

    /// Returns true if there are unfinished requests, or finished requests
    /// not yet returned in SchedulerOutputs.
    fn has_requests(&self) -> bool {
        self.has_unfinished_requests() || self.has_finished_requests()
    }

    /// Reset the prefix cache for KV cache.
    ///
    /// This is particularly required when the model weights are live-updated.
    fn reset_prefix_cache(&mut self) -> bool;

    /// Returns (num_running_reqs, num_waiting_reqs).
    fn get_request_counts(&self) -> (usize, usize);

    /// Make a SchedulerStats object for logging.
    ///
    /// The SchedulerStats object is created for every scheduling step.
    fn make_stats(&mut self) -> Option<SchedulerStats>;

    /// Shutdown the scheduler.
    fn shutdown(&mut self);
}

