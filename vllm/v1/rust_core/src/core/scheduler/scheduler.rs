use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::core::{
    CachedRequestData, KVCacheBlocks, KVCacheConfig, KVCacheManager, NewRequestData, Request,
    RequestStatus, SchedulerOutput,
};

use crate::core::types::{
    engine_core_outputs::{EngineCoreOutput, EngineCoreOutputs, SchedulerStats},
    model_runner_output::{DraftTokenIds, ModelRunnerOutput},
};

use super::encoder_cache_manager::EncoderCacheManager;
use super::request_queue::{create_request_queue, RequestQueue, SchedulingPolicy};
use super::scheduler_interface::SchedulerInterface;
use super::scheduler_utils::{calculate_chunked_prefill_tokens, check_stop, TokenBudget};

/// Helper function to remove requests from a vector based on request IDs
fn remove_requests_by_ids(
    vec: Vec<Rc<RefCell<Request>>>,
    to_remove: &HashSet<String>,
) -> Vec<Rc<RefCell<Request>>> {
    vec.into_iter()
        .filter(|req| !to_remove.contains(&req.borrow().request_id))
        .collect()
}

/// Configuration for the scheduler
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_model_len: usize,
    pub max_num_encoder_input_tokens: usize,
    pub policy: String,
    pub long_prefill_token_threshold: usize,
    pub chunked_prefill_enabled: bool,
    pub disable_chunked_mm_input: bool,
}

/// The main scheduler implementation
pub struct Scheduler {
    // Configuration
    scheduler_config: SchedulerConfig,
    kv_cache_config: KVCacheConfig,
    enable_prefix_caching: bool,
    log_stats: bool,
    is_encoder_decoder: bool,
    use_pp: bool,
    block_size: usize,

    // Scheduling constraints
    max_num_running_reqs: usize,
    max_num_scheduled_tokens: usize,
    max_model_len: usize,
    max_num_encoder_input_tokens: usize,

    // Scheduling policy
    policy: SchedulingPolicy,

    // Request management
    requests: HashMap<String, Rc<RefCell<Request>>>,
    waiting: Box<dyn RequestQueue>,
    running: Vec<Rc<RefCell<Request>>>,
    finished_req_ids: HashSet<String>,
    finished_req_ids_dict: Option<HashMap<i32, HashSet<String>>>,

    // KV cache management
    kv_cache_manager: KVCacheManager,
    encoder_cache_manager: EncoderCacheManager,

    // Speculative decoding
    use_eagle: bool,
    num_spec_tokens: usize,
    num_lookahead_tokens: usize,

    // LoRA state
    scheduled_loras: HashSet<i32>,
    max_loras: Option<usize>,

    // KV Connector state (for distributed caching)
    finished_recving_kv_req_ids: HashSet<String>,
    use_kv_connector: bool,
}

impl Scheduler {
    pub fn new(
        scheduler_config: SchedulerConfig,
        kv_cache_config: KVCacheConfig,
        enable_prefix_caching: bool,
        include_finished_set: bool,
        log_stats: bool,
        is_encoder_decoder: bool,
        use_pp: bool,
    ) -> Self {
        // Determine scheduling policy
        let policy = match scheduler_config.policy.as_str() {
            "priority" => SchedulingPolicy::Priority,
            "fcfs" => SchedulingPolicy::Fcfs,
            _ => panic!("Unknown scheduling policy: {}", scheduler_config.policy),
        };

        // Create request queue based on policy
        let waiting = create_request_queue(policy);

        // Initialize finished_req_ids_dict if needed
        let finished_req_ids_dict = if include_finished_set {
            Some(HashMap::new())
        } else {
            None
        };

        // Create KV cache manager
        let kv_cache_manager = KVCacheManager::new(
            kv_cache_config.clone(),
            scheduler_config.max_model_len,
            enable_prefix_caching,
            false, // use_eagle - will be set based on speculative config
            log_stats,
            false, // enable_kv_cache_events - simplified for now
            1,     // dcp_world_size - simplified for now
        );

        // Create encoder cache manager
        let encoder_cache_manager =
            EncoderCacheManager::new(scheduler_config.max_num_encoder_input_tokens);

        let max_num_running_reqs = scheduler_config.max_num_seqs;
        let max_num_scheduled_tokens = scheduler_config.max_num_batched_tokens;
        let max_model_len = scheduler_config.max_model_len;
        let max_num_encoder_input_tokens = scheduler_config.max_num_encoder_input_tokens;
        // TODO: Get block size from KVCacheSpec once available
        let block_size = 16;

        Self {
            scheduler_config,
            kv_cache_config,
            enable_prefix_caching,
            log_stats,
            is_encoder_decoder,
            use_pp,
            block_size,
            max_num_running_reqs,
            max_num_scheduled_tokens,
            max_model_len,
            max_num_encoder_input_tokens,
            policy,
            requests: HashMap::new(),
            waiting,
            running: Vec::new(),
            finished_req_ids: HashSet::new(),
            finished_req_ids_dict,
            kv_cache_manager,
            encoder_cache_manager,
            use_eagle: false,
            num_spec_tokens: 0,
            num_lookahead_tokens: 0,
            scheduled_loras: HashSet::new(),
            max_loras: None, // TODO: Get from config when LoRA config is added
            finished_recving_kv_req_ids: HashSet::new(),
            use_kv_connector: false, // TODO: Set based on config
        }
    }

    /// Main scheduling algorithm - determines which requests to process
    fn schedule_impl(&mut self) -> SchedulerOutput {
        let mut scheduled_new_reqs: Vec<Rc<RefCell<Request>>> = Vec::new();
        let mut scheduled_resumed_reqs: Vec<Rc<RefCell<Request>>> = Vec::new();
        let mut scheduled_running_reqs: Vec<Rc<RefCell<Request>>> = Vec::new();
        let mut preempted_reqs: Vec<Rc<RefCell<Request>>> = Vec::new();

        let mut req_to_new_blocks: HashMap<String, KVCacheBlocks> = HashMap::new();
        let mut num_scheduled_tokens: HashMap<String, usize> = HashMap::new();
        let mut token_budget = TokenBudget::new(self.max_num_scheduled_tokens);

        // Encoder-related
        let mut scheduled_encoder_inputs: HashMap<String, Vec<usize>> = HashMap::new();
        let mut encoder_compute_budget = self.max_num_encoder_input_tokens;

        // Speculative decoding related
        let mut scheduled_spec_decode_tokens: HashMap<String, Vec<i32>> = HashMap::new();

        // First, schedule the RUNNING requests
        let mut req_index = 0;
        while req_index < self.running.len() && token_budget.has_tokens() {
            let request = self.running[req_index].clone();
            let req = request.borrow_mut();

            let mut num_new_tokens = (req.num_tokens_with_spec() + req.num_output_placeholders)
                .saturating_sub(req.num_computed_tokens);

            // Apply long prefill threshold if configured
            if self.scheduler_config.long_prefill_token_threshold > 0
                && num_new_tokens > self.scheduler_config.long_prefill_token_threshold
            {
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold;
            }

            // Respect token budget
            num_new_tokens = token_budget.allocate(num_new_tokens);

            // Make sure input position doesn't exceed max model len
            num_new_tokens = num_new_tokens.min(self.max_model_len - 1 - req.num_computed_tokens);

            // Schedule encoder inputs if needed
            let encoder_inputs_to_schedule;
            let new_encoder_compute_budget;
            drop(req); // Release borrow for try_schedule_encoder_inputs
            let req_ref = request.borrow();
            (
                encoder_inputs_to_schedule,
                num_new_tokens,
                new_encoder_compute_budget,
            ) = self.try_schedule_encoder_inputs(
                &*req_ref,
                req_ref.num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
            );
            drop(req_ref);
            let mut req = request.borrow_mut();

            if num_new_tokens == 0 {
                req_index += 1;
                continue;
            }

            // Handle speculative decoding tokens if present
            if !req.spec_token_ids.is_empty() {
                let num_scheduled_spec_tokens =
                    (num_new_tokens + req.num_computed_tokens).saturating_sub(req.num_tokens());
                if num_scheduled_spec_tokens > 0 {
                    // Trim spec_token_ids to num_scheduled_spec_tokens
                    let spec_tokens = if req.spec_token_ids.len() > num_scheduled_spec_tokens {
                        // Keep only the first num_scheduled_spec_tokens
                        let kept_tokens = req.spec_token_ids[..num_scheduled_spec_tokens].to_vec();
                        // Remove the scheduled tokens from the request
                        req.spec_token_ids.drain(..num_scheduled_spec_tokens);
                        kept_tokens
                    } else {
                        // Use all spec tokens and clear the list
                        let all_tokens = req.spec_token_ids.clone();
                        req.spec_token_ids.clear();
                        all_tokens
                    };
                    scheduled_spec_decode_tokens.insert(req.request_id.clone(), spec_tokens);
                }
            }
            drop(req);

            // Try to allocate KV cache blocks
            let req = request.borrow();
            let new_blocks = self.kv_cache_manager.allocate_slots(
                &*req,
                num_new_tokens,
                0,    // num_new_local_computed_tokens
                None, // new_computed_blocks
                self.num_lookahead_tokens,
                false, // delay_cache_blocks
                0,     // num_encoder_tokens
            );

            let can_schedule = if let Some(blocks) = new_blocks {
                // Successfully allocated blocks
                drop(req);
                scheduled_running_reqs.push(request.clone());
                let request_id = request.borrow().request_id.clone();
                req_to_new_blocks.insert(request_id.clone(), blocks);
                num_scheduled_tokens.insert(request_id.clone(), num_new_tokens);

                // Save encoder inputs and update budget
                if !encoder_inputs_to_schedule.is_empty() {
                    scheduled_encoder_inputs
                        .insert(request_id.clone(), encoder_inputs_to_schedule.clone());
                    // Allocate the encoder cache
                    for i in encoder_inputs_to_schedule {
                        self.encoder_cache_manager.allocate(&*request.borrow(), i);
                    }
                    encoder_compute_budget = new_encoder_compute_budget;
                }
                true
            } else {
                // Need to preempt a request
                drop(req);
                if self.preempt_request(&mut preempted_reqs) {
                    // Try again after preemption
                    continue;
                } else {
                    // No more requests to preempt
                    false
                }
            };

            if !can_schedule {
                break;
            }

            req_index += 1;
        }

        // Create a queue for skipped requests that we'll add back later
        let mut skipped_waiting_requests = create_request_queue(self.policy.clone());

        // Next, schedule the WAITING requests
        if preempted_reqs.is_empty() {
            while !self.waiting.is_empty() && token_budget.has_tokens() {
                if self.running.len() >= self.max_num_running_reqs {
                    break;
                }

                let request = self.waiting.peek_request();
                if request.is_none() {
                    break;
                }
                let request = request.expect("peek_request returned Some but unwrap failed");

                let req = request.borrow();

                // Handle special waiting states
                if req.status == RequestStatus::WaitingForRemoteKvs {
                    // Check if KV transfer is complete
                    if self.use_kv_connector
                        && self.finished_recving_kv_req_ids.contains(&req.request_id)
                    {
                        // KV transfer is complete, can proceed with scheduling
                        self.finished_recving_kv_req_ids.remove(&req.request_id);
                        // The request will be processed below
                    } else {
                        // Still waiting for KV transfer
                        drop(req);
                        let skipped_request = self.waiting.pop_request()
                            .expect("request disappeared between peek and pop");
                        skipped_waiting_requests.prepend_request(skipped_request);
                        continue;
                    }
                } else if req.status == RequestStatus::WaitingForFsm {
                    // Skip FSM waiting requests
                    drop(req);
                    let skipped_request = self.waiting.pop_request()
                        .expect("request disappeared between peek and pop");
                    skipped_waiting_requests.prepend_request(skipped_request);
                    continue;
                }

                // Check LoRA constraints if max_loras is set
                if let Some(max_loras) = self.max_loras {
                    if let Some(lora_request) = &req.lora_request {
                        // Check if this LoRA is already scheduled or if we have room
                        if !self.scheduled_loras.contains(&lora_request.lora_int_id)
                            && self.scheduled_loras.len() >= max_loras
                        {
                            // Can't schedule this request - too many LoRAs
                            drop(req);
                            break;
                        }
                    }
                }

                let num_computed_tokens = req.num_computed_tokens;
                let mut num_new_tokens = req.num_tokens() - num_computed_tokens;

                // Check if we can fit this request
                if !self.scheduler_config.chunked_prefill_enabled
                    && num_new_tokens > token_budget.remaining
                {
                    drop(req);
                    break;
                }

                num_new_tokens = calculate_chunked_prefill_tokens(
                    num_new_tokens,
                    self.scheduler_config.long_prefill_token_threshold,
                    token_budget.remaining,
                );

                // Schedule encoder inputs if needed
                let encoder_inputs_to_schedule;
                let new_encoder_compute_budget;
                drop(req); // Release borrow for try_schedule_encoder_inputs
                let req_ref = request.borrow();
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                ) = self.try_schedule_encoder_inputs(
                    &*req_ref,
                    req_ref.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                );
                drop(req_ref);
                let req = request.borrow();

                if num_new_tokens == 0 {
                    drop(req);
                    break;
                }

                // Try to allocate blocks
                let new_blocks = self.kv_cache_manager.allocate_slots(
                    &*req,
                    num_new_tokens,
                    0,    // num_new_local_computed_tokens
                    None, // new_computed_blocks
                    if num_computed_tokens == 0 {
                        self.num_lookahead_tokens
                    } else {
                        0
                    },
                    false, // delay_cache_blocks
                    0,     // num_encoder_tokens
                );

                if new_blocks.is_none() {
                    drop(req);
                    break;
                }

                // Successfully scheduled
                drop(req);
                let request = self.waiting.pop_request()
                    .expect("request disappeared between peek and pop");

                {
                    let mut req = request.borrow_mut();
                    let was_waiting = req.status == RequestStatus::Waiting;
                    let was_preempted = req.status == RequestStatus::Preempted;

                    req.status = RequestStatus::Running;
                    req.num_computed_tokens = num_computed_tokens;

                    // Update scheduled LoRAs if this request has a LoRA
                    if let Some(lora_request) = &req.lora_request {
                        self.scheduled_loras.insert(lora_request.lora_int_id);
                    }

                    if was_waiting {
                        scheduled_new_reqs.push(request.clone());
                    } else if was_preempted {
                        scheduled_resumed_reqs.push(request.clone());
                    }
                }

                self.running.push(request.clone());
                let request_id = request.borrow().request_id.clone();
                req_to_new_blocks.insert(request_id.clone(), new_blocks
                    .expect("new_blocks is None after allocation check"));
                num_scheduled_tokens.insert(request_id.clone(), num_new_tokens);
                token_budget.allocate(num_new_tokens);

                // Save encoder inputs and update budget
                if !encoder_inputs_to_schedule.is_empty() {
                    scheduled_encoder_inputs
                        .insert(request_id.clone(), encoder_inputs_to_schedule.clone());
                    // Allocate the encoder cache
                    for i in encoder_inputs_to_schedule {
                        self.encoder_cache_manager.allocate(&*request.borrow(), i);
                    }
                    encoder_compute_budget = new_encoder_compute_budget;
                }
            }
        }

        // Put back any skipped requests at the head of the waiting queue
        if !skipped_waiting_requests.is_empty() {
            // Drain all skipped requests and prepend them to waiting queue
            let mut skipped_reqs = Vec::new();
            while let Some(req) = skipped_waiting_requests.pop_request() {
                skipped_reqs.push(req);
            }
            // Reverse to maintain original order when prepending
            skipped_reqs.reverse();
            self.waiting.prepend_requests(&skipped_reqs);
        }

        // Construct scheduler output
        let new_reqs_data: Vec<NewRequestData> = scheduled_new_reqs
            .iter()
            .map(|req| {
                let r = req.borrow();
                let blocks = req_to_new_blocks.get(&r.request_id)
                    .expect("request blocks not found in req_to_new_blocks");
                NewRequestData::from_request(&*r, blocks.get_block_ids())
            })
            .collect();

        let cached_reqs_data = self.make_cached_request_data(
            &scheduled_running_reqs,
            &scheduled_resumed_reqs,
            &num_scheduled_tokens,
            &req_to_new_blocks,
        );

        let total_num_scheduled_tokens = num_scheduled_tokens.values().sum();

        SchedulerOutput {
            scheduled_new_reqs: new_reqs_data,
            scheduled_cached_reqs: cached_reqs_data,
            num_scheduled_tokens: num_scheduled_tokens.clone(),
            total_num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            scheduled_encoder_inputs: scheduled_encoder_inputs,
            num_common_prefix_blocks: vec![0; self.kv_cache_config.kv_cache_groups.len()],
            finished_req_ids: self.finished_req_ids.clone(),
            free_encoder_mm_hashes: Vec::new(),
            structured_output_request_ids: HashMap::new(),
            grammar_bitmask: None,
            kv_connector_metadata: None,
        }
    }

    /// Preempt the lowest priority request
    fn preempt_request(&mut self, preempted_reqs: &mut Vec<Rc<RefCell<Request>>>) -> bool {
        if self.running.is_empty() {
            return false;
        }

        // Find the request to preempt based on policy
        let preempted_idx = if self.policy == SchedulingPolicy::Priority {
            // Find request with highest priority value (lowest priority)
            self.running
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let ra = a.borrow();
                    let rb = b.borrow();
                    // Compare by priority first, then by arrival_time
                    match ra.priority.cmp(&rb.priority) {
                        std::cmp::Ordering::Equal => {
                            // Use partial_cmp for f64 and unwrap_or Equal
                            ra.arrival_time
                                .partial_cmp(&rb.arrival_time)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        }
                        other => other,
                    }
                })
                .map(|(idx, _)| idx)
                .unwrap_or(self.running.len() - 1)
        } else {
            // FCFS: preempt the last request
            self.running.len() - 1
        };

        let preempted_req = self.running.remove(preempted_idx);

        {
            let mut req = preempted_req.borrow_mut();
            self.kv_cache_manager.free(&*req);
            self.encoder_cache_manager.free(&*req);
            req.status = RequestStatus::Preempted;
            req.num_computed_tokens = 0;

            // Remove LoRA from scheduled set if no other running requests use it
            if let Some(lora_request) = &req.lora_request {
                let lora_int_id = lora_request.lora_int_id;
                let mut lora_still_used = false;
                for other_req in &self.running {
                    let other = other_req.borrow();
                    if let Some(other_lora) = &other.lora_request {
                        if other_lora.lora_int_id == lora_int_id {
                            lora_still_used = true;
                            break;
                        }
                    }
                }
                if !lora_still_used {
                    self.scheduled_loras.remove(&lora_int_id);
                }
            }
        }

        self.waiting.prepend_request(preempted_req.clone());
        preempted_reqs.push(preempted_req);

        true
    }

    /// Create cached request data for running and resumed requests
    fn make_cached_request_data(
        &self,
        running_reqs: &[Rc<RefCell<Request>>],
        resumed_reqs: &[Rc<RefCell<Request>>],
        num_scheduled_tokens: &HashMap<String, usize>,
        req_to_new_blocks: &HashMap<String, KVCacheBlocks>,
    ) -> CachedRequestData {
        let mut req_ids = Vec::new();
        let mut new_token_ids = Vec::new();
        let mut new_block_ids = Vec::new();
        let mut num_computed_tokens_list = Vec::new();
        let mut resumed_from_preemption = Vec::new();

        // Process running requests
        for req in running_reqs {
            let r = req.borrow();
            req_ids.push(r.request_id.clone());

            let num_tokens = *num_scheduled_tokens.get(&r.request_id).unwrap_or(&0);

            if self.use_pp {
                // When using PP, include the sampled tokens
                let start = r.num_computed_tokens;
                let end = start + num_tokens;
                let tokens: Vec<i32> = r.all_token_ids[start..end].to_vec();
                new_token_ids.push(tokens);
            } else {
                new_token_ids.push(Vec::new());
            }

            let blocks = req_to_new_blocks.get(&r.request_id);
            new_block_ids.push(blocks.map(|b| b.get_block_ids()));

            num_computed_tokens_list.push(r.num_computed_tokens);
            resumed_from_preemption.push(false);
        }

        // Process resumed requests
        for req in resumed_reqs {
            let r = req.borrow();
            req_ids.push(r.request_id.clone());

            let num_tokens = *num_scheduled_tokens.get(&r.request_id).unwrap_or(&0);

            if self.use_pp {
                let start = r.num_computed_tokens;
                let end = start + num_tokens;
                let tokens: Vec<i32> = r.all_token_ids[start..end].to_vec();
                new_token_ids.push(tokens);
            } else {
                new_token_ids.push(Vec::new());
            }

            let blocks = req_to_new_blocks.get(&r.request_id);
            new_block_ids.push(blocks.map(|b| b.get_block_ids()));

            num_computed_tokens_list.push(r.num_computed_tokens);
            resumed_from_preemption.push(true);
        }

        CachedRequestData {
            req_ids,
            resumed_from_preemption,
            new_token_ids,
            new_block_ids,
            num_computed_tokens: num_computed_tokens_list,
        }
    }

    /// Update request after schedule
    fn update_after_schedule(&mut self, scheduler_output: &SchedulerOutput) {
        // Advance the number of computed tokens for scheduled requests
        for (req_id, num_tokens) in &scheduler_output.num_scheduled_tokens {
            if let Some(request) = self.requests.get(req_id) {
                let mut req = request.borrow_mut();
                req.num_computed_tokens += num_tokens;
            }
        }

        // Clear finished request IDs (but keep reference for output)
        self.finished_req_ids.clear();
    }

    /// Update request with new output tokens
    fn update_request_with_output(
        &mut self,
        request: &mut Request,
        new_token_ids: Vec<i32>,
    ) -> (Vec<i32>, bool) {
        // Add new tokens to request
        for token_id in &new_token_ids {
            request.append_output_token_id(*token_id);
        }

        // Check for stop conditions
        let stopped = check_stop(request, self.max_model_len, None);

        (new_token_ids, stopped)
    }

    /// Free a finished request and potentially return KV transfer parameters
    fn free_request(&mut self, request: &Request) -> Option<HashMap<String, String>> {
        assert!(request.is_finished());

        // Handle KV Connector if enabled
        let kv_transfer_params = if self.use_kv_connector {
            // In a real implementation, this would interact with the KV connector
            // to determine if we need to delay freeing blocks and get transfer params
            // For now, we'll just return None
            None
        } else {
            None
        };

        self.encoder_cache_manager.free(request);
        let request_id = request.request_id.clone();
        self.finished_req_ids.insert(request_id.clone());

        if let Some(finished_dict) = &mut self.finished_req_ids_dict {
            finished_dict
                .entry(request.client_index as i32)
                .or_insert_with(HashSet::new)
                .insert(request_id.clone());
        }

        // Remove LoRA from scheduled set if this request had one
        if let Some(lora_request) = &request.lora_request {
            // Check if any other running requests use this LoRA
            let mut lora_still_used = false;
            for running_req in &self.running {
                let req = running_req.borrow();
                if req.request_id != request.request_id {
                    if let Some(other_lora) = &req.lora_request {
                        if other_lora.lora_int_id == lora_request.lora_int_id {
                            lora_still_used = true;
                            break;
                        }
                    }
                }
            }
            if !lora_still_used {
                self.scheduled_loras.remove(&lora_request.lora_int_id);
            }
        }

        // Free blocks if not delayed by KV connector
        if kv_transfer_params.is_none() {
            self.kv_cache_manager.free(request);
            self.requests.remove(&request_id);
        }

        kv_transfer_params
    }

    /// Mark requests that have finished receiving KV transfers
    pub fn mark_kv_transfers_finished(&mut self, request_ids: Vec<String>) {
        for request_id in request_ids {
            self.finished_recving_kv_req_ids.insert(request_id);
        }
    }

    /// Update draft token ids for speculative decoding
    pub fn update_draft_token_ids(&mut self, req_ids: Vec<String>, draft_token_ids: Vec<Vec<i32>>) {
        for (req_id, spec_token_ids) in req_ids.iter().zip(draft_token_ids.iter()) {
            if let Some(request) = self.requests.get(req_id) {
                let mut req = request.borrow_mut();
                if req.is_finished() {
                    continue;
                }

                // Update spec_token_ids
                if spec_token_ids.is_empty() {
                    req.spec_token_ids.clear();
                } else {
                    // In a real implementation, we might validate these tokens
                    // against structured output constraints
                    req.spec_token_ids = spec_token_ids.clone();
                }
            }
        }
    }

    /// Determine which encoder inputs need to be scheduled in the current step
    fn try_schedule_encoder_inputs(
        &mut self,
        request: &Request,
        num_computed_tokens: usize,
        mut num_new_tokens: usize,
        mut encoder_compute_budget: usize,
    ) -> (Vec<usize>, usize, usize) {
        if num_new_tokens == 0 || !request.has_encoder_inputs {
            return (Vec::new(), num_new_tokens, encoder_compute_budget);
        }

        let mut encoder_inputs_to_schedule = Vec::new();
        let mut mm_hashes_to_schedule = std::collections::HashSet::new();
        let mut num_tokens_to_schedule = 0;

        for (i, mm_feature) in request.mm_features.iter().enumerate() {
            let start_pos = mm_feature.mm_position.offset;
            let num_encoder_tokens = mm_feature.mm_position.length;

            // The encoder output is needed if the two ranges overlap:
            // [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            // [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens {
                // The encoder input is not needed in this step
                break;
            }

            if self.is_encoder_decoder && num_computed_tokens > 0 {
                assert_eq!(
                    start_pos, 0,
                    "Encoder input should be processed at the beginning of \
                     the sequence when encoder-decoder models are used."
                );
                // Encoder input has already been computed
                continue;
            } else if start_pos + num_encoder_tokens <= num_computed_tokens {
                // The encoder input is already computed and stored
                // in the decoder's KV cache
                continue;
            }

            if !self.is_encoder_decoder {
                // We are not using the encoder cache for encoder-decoder models yet
                if mm_hashes_to_schedule.contains(&mm_feature.identifier) {
                    // The same encoder input has already been scheduled
                    continue;
                }

                if self
                    .encoder_cache_manager
                    .check_and_update_cache(request, i)
                {
                    // The encoder input is already computed and cached
                    continue;
                }
            }

            // If no encoder input chunking is allowed, we do not want to
            // partially schedule a multimodal item
            if self.scheduler_config.disable_chunked_mm_input
                && num_computed_tokens < start_pos
                && (num_computed_tokens + num_new_tokens) < (start_pos + num_encoder_tokens)
            {
                num_new_tokens = start_pos - num_computed_tokens;
                break;
            }

            if !self.encoder_cache_manager.can_allocate(
                request,
                i,
                encoder_compute_budget,
                num_tokens_to_schedule,
            ) {
                // The encoder cache is full or the encoder budget is exhausted
                if num_computed_tokens < start_pos {
                    // We only schedule the decoder tokens just before the encoder input
                    num_new_tokens = start_pos - num_computed_tokens;
                } else {
                    // Because of prefix caching, num_computed_tokens is greater
                    // than start_pos even though its encoder input is not
                    // available. In this case, we can't schedule any token
                    num_new_tokens = 0;
                }
                break;
            }

            num_tokens_to_schedule += num_encoder_tokens;
            encoder_compute_budget -= num_encoder_tokens;
            mm_hashes_to_schedule.insert(mm_feature.identifier.clone());
            encoder_inputs_to_schedule.push(i);
        }

        (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
        )
    }
}

impl SchedulerInterface for Scheduler {
    fn schedule(&mut self) -> SchedulerOutput {
        let output = self.schedule_impl();
        self.update_after_schedule(&output);
        output
    }

    fn update_from_output(
        &mut self,
        scheduler_output: &SchedulerOutput,
        model_runner_output: &ModelRunnerOutput,
    ) -> HashMap<i32, EngineCoreOutputs> {
        let mut outputs: HashMap<i32, Vec<EngineCoreOutput>> = HashMap::new();

        let mut stopped_running_req_ids: HashSet<String> = HashSet::new();
        let mut stopped_preempted_reqs: Vec<Rc<RefCell<Request>>> = Vec::new();

        // Process each scheduled request
        for (req_id, _num_tokens_scheduled) in &scheduler_output.num_scheduled_tokens {
            let request = match self.requests.get(req_id) {
                Some(req) => req.clone(),
                None => continue, // Request already finished
            };

            let req_index = match model_runner_output.req_id_to_index.get(req_id) {
                Some(idx) => *idx,
                None => continue,
            };

            let generated_token_ids = if !model_runner_output.sampled_token_ids.is_empty() {
                model_runner_output.sampled_token_ids[req_index].clone()
            } else {
                Vec::new()
            };

            let mut req = request.borrow_mut();
            let status_before_stop = req.status;

            // Update request with new tokens and check for stop
            let (new_token_ids, stopped) = if !generated_token_ids.is_empty() {
                self.update_request_with_output(&mut req, generated_token_ids)
            } else {
                (Vec::new(), false)
            };

            // Check for pooler output stop condition
            let pooler_output = model_runner_output.pooler_output.as_ref();
            let stopped = stopped || check_stop(&mut req, self.max_model_len, pooler_output);

            if stopped {
                let req_id_clone = req.request_id.clone();
                drop(req);
                self.free_request(&request.borrow());

                if status_before_stop == RequestStatus::Running {
                    stopped_running_req_ids.insert(req_id_clone);
                } else {
                    stopped_preempted_reqs.push(request.clone());
                }
            }

            // Create output for this request if needed
            if !new_token_ids.is_empty() || stopped {
                let req = request.borrow();
                let output = EngineCoreOutput {
                    request_id: req_id.clone(),
                    new_token_ids,
                    finish_reason: if stopped {
                        req.get_finished_reason()
                    } else {
                        None
                    },
                    new_logprobs: None,
                    new_prompt_logprobs_tensors: None,
                    pooling_output: None,
                    stop_reason: req.stop_reason.clone(),
                    events: Vec::new(),
                    kv_transfer_params: None,
                    trace_headers: None,
                    num_cached_tokens: req.num_cached_tokens,
                };

                outputs
                    .entry(req.client_index as i32)
                    .or_insert_with(Vec::new)
                    .push(output);
            }
        }

        // Remove stopped requests from running queue
        if !stopped_running_req_ids.is_empty() {
            self.running = remove_requests_by_ids(self.running.clone(), &stopped_running_req_ids);
        }

        // Remove stopped requests from waiting queue
        if !stopped_preempted_reqs.is_empty() {
            self.waiting.remove_requests(&stopped_preempted_reqs);
        }

        // Convert to EngineCoreOutputs
        outputs
            .into_iter()
            .map(|(client_index, outs)| {
                (
                    client_index,
                    EngineCoreOutputs {
                        outputs: outs,
                        scheduler_stats: None,
                        finished_requests: Vec::new(),
                        wave_complete: false,
                    },
                )
            })
            .collect()
    }

    fn update_draft_token_ids(&mut self, draft_token_ids: &DraftTokenIds) {
        for (req_id, spec_token_ids) in draft_token_ids
            .req_ids
            .iter()
            .zip(draft_token_ids.draft_token_ids.iter())
        {
            if let Some(request) = self.requests.get(req_id) {
                let mut req = request.borrow_mut();
                if !req.is_finished() {
                    req.spec_token_ids = spec_token_ids.clone();
                }
            }
        }
    }

    fn add_request(&mut self, request: Request) {
        let req_rc = Rc::new(RefCell::new(request));
        let req_id = req_rc.borrow().request_id.clone();

        self.waiting.add_request(req_rc.clone());
        self.requests.insert(req_id, req_rc);
    }

    fn finish_requests(&mut self, request_ids: Vec<String>, finished_status: RequestStatus) {
        assert!(finished_status.is_finished());

        let mut running_req_ids_to_remove = HashSet::new();
        let mut waiting_requests_to_remove = Vec::new();
        let mut valid_requests = Vec::new();

        // First pass: collect requests to remove
        for req_id in &request_ids {
            if let Some(request) = self.requests.get(req_id) {
                let req = request.borrow();
                valid_requests.push(request.clone());

                if req.status == RequestStatus::Running {
                    running_req_ids_to_remove.insert(req_id.clone());
                } else {
                    waiting_requests_to_remove.push(request.clone());
                }
            }
        }

        // Remove from queues
        if !running_req_ids_to_remove.is_empty() {
            self.running = remove_requests_by_ids(self.running.clone(), &running_req_ids_to_remove);
        }
        if !waiting_requests_to_remove.is_empty() {
            self.waiting.remove_requests(&waiting_requests_to_remove);
        }

        // Second pass: set status and free
        for request in valid_requests {
            {
                let mut req = request.borrow_mut();
                req.status = finished_status;
            }
            self.free_request(&request.borrow());
        }
    }

    fn get_num_unfinished_requests(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    fn has_finished_requests(&self) -> bool {
        !self.finished_req_ids.is_empty()
    }

    fn reset_prefix_cache(&mut self) -> bool {
        self.kv_cache_manager.reset_prefix_cache()
    }

    fn get_request_counts(&self) -> (usize, usize) {
        (self.running.len(), self.waiting.len())
    }

    fn make_stats(&mut self) -> Option<SchedulerStats> {
        if !self.log_stats {
            return None;
        }

        let prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats();

        Some(SchedulerStats {
            num_running_reqs: self.running.len(),
            num_waiting_reqs: self.waiting.len(),
            kv_cache_usage: self.kv_cache_manager.usage(),
            prefix_cache_stats: prefix_cache_stats.unwrap_or_default(),
            spec_decoding_stats: None,
            num_corrupted_reqs: 0,
        })
    }

    fn shutdown(&mut self) {
        // Clean up any resources
        // In the Python version, this shuts down event publishers and connectors
        // For now, we just clear our data structures
        self.running.clear();
        // Clear waiting queue by draining it
        while self.waiting.pop_request().is_some() {}
        self.requests.clear();
        self.finished_req_ids.clear();
        if let Some(dict) = &mut self.finished_req_ids_dict {
            dict.clear();
        }
    }
}
