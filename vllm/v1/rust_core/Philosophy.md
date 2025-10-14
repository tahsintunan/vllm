# Philosophy
The Rust Core is focused on the performance-critical parts:
- Block allocation/deallocation (CPU-bound)
- Request scheduling decisions (CPU-bound)
- KV cache management (CPU-bound)

While keeping domain-specific logic in Python:
- Grammar/FSM validation (complex, changes frequently)
- LoRA adapter management details
- Model-specific logic

The `accept_tokens()` call happens in update_from_output() - this is AFTER the model runs and generates tokens. The sequence is:

1. Rust Scheduler decides which requests to run → ScheduleOutput
2. GPU/Model generates tokens
3. Python processes the generated tokens (including grammar validation)
4. Python calls back to Rust Scheduler with updates

So the Rust Core doesn't need to handle grammar - it just needs to:
- Know that a request uses structured output (to set correct status)
- Return that info in SchedulerOutput.structured_output_request_ids
- Let Python handle the grammar validation after tokens are generated

# Data Flow Architecture

Initial Request Submission (Python → Rust)

# Python
python_request = Request(
    request_id="req-1",
    prompt_token_ids=[1, 2, 3, ...],
    sampling_params=SamplingParams(...),
    ...
)

# Convert to minimal scheduling data (NOT full Request copy!)
scheduling_data = {
    'request_id': python_request.request_id,
    'num_tokens': len(python_request.all_token_ids),
    'max_tokens': python_request.max_tokens,
    'priority': python_request.priority,
    'has_encoder_inputs': bool(python_request.mm_features),
    # Only what scheduler needs
}

rust_scheduler.add_request(scheduling_data)  # FFI call

Scheduling Phase (Rust → Python)

// Rust returns lightweight scheduling decisions
pub struct ScheduleOutput {
    pub scheduled_new_reqs: Vec<String>,  // Just IDs!
    pub scheduled_running_reqs: Vec<String>,
    pub preempted_reqs: Vec<String>,
    pub blocks_to_allocate: HashMap<String, Vec<u32>>,
    pub blocks_to_free: Vec<u32>,
    // No token data, no full requests
}

Token Generation Phase (Python + GPU)

# Python uses schedule output to prepare GPU batch
schedule_output = rust_scheduler.schedule()

# Python still has all the original Request objects
gpu_batch = []
for req_id in schedule_output.scheduled_running_reqs:
    req = python_requests[req_id]  # Original Python Request
    gpu_batch.append({
        'tokens': req.all_token_ids,
        'blocks': schedule_output.blocks_to_allocate[req_id],
        # ...
    })

# GPU generates tokens
new_tokens = gpu_model.generate(gpu_batch)

Update Phase (Python → Rust)

# Update Rust with token generation results
for req_id, tokens in new_tokens.items():
    python_request = python_requests[req_id]

    # Python handles complex logic
    if python_request.structured_output_request:
        python_request.structured_output_request.grammar.accept_tokens(tokens)

    # Rust only gets simple updates
    rust_scheduler.update_request(req_id, len(tokens))  # Just the count!

Number of Round Trips

Per scheduling iteration:
1. Schedule call: Python → Rust → Python (get scheduling decisions)
2. Update calls: Python → Rust (update token counts after generation)

So 2 FFI crossings per iteration, not per request!

Overhead Mitigation Strategies

1. Don't Copy Full Requests

// BAD: Full request copy
struct Request {
    prompt_token_ids: Vec<i32>,  // Could be thousands of tokens!
    output_token_ids: Vec<i32>,  // Growing array
    // ... many fields
}

// GOOD: Scheduling view
struct SchedulingRequest {
    request_id: String,
    num_tokens: usize,
    max_tokens: u32,
    priority: i32,
    // Only scheduling-relevant fields
}

2. Batch Operations

# BAD: Individual calls
for req in requests:
    rust_scheduler.add_request(req)  # N FFI calls

# GOOD: Batch call
rust_scheduler.add_requests(requests)  # 1 FFI call

3. ID-Based References

// Rust tracks requests by ID, Python keeps the actual data
pub struct Scheduler {
    requests: HashMap<String, SchedulingMetadata>,  // Lightweight!
}

4. Lazy Updates

# Only send updates that affect scheduling
if tokens_generated > 0 or request_finished:
    rust_scheduler.update_request(req_id, update_data)
# Don't update if nothing scheduling-relevant changed

Memory Architecture

Python Process Memory          Rust Shared Library Memory
┌─────────────────────┐       ┌──────────────────────┐
│ Full Request Objects│       │ Scheduling Metadata  │
│ - Token arrays      │       │ - Request IDs        │
│ - Grammar state     │  ←──→ │ - Block assignments  │
│ - LoRA configs      │  FFI  │ - Queue positions    │
│ - MM features       │       │ - Token counts only  │
└─────────────────────┘       └──────────────────────┘

Performance Estimate

The overhead should be minimal because:
1. No token array copying - Just counts and IDs cross FFI
2. Batch operations - One schedule call handles many requests
3. Rust owns block management - The most CPU-intensive part stays in Rust
4. Python keeps complex objects - No serialization of grammar/LoRA/etc.

The FFI overhead (~100ns per call) is negligible compared to:
- Model inference time (milliseconds)
- Block allocation algorithms (microseconds)
- Python's current pure-Python scheduler (microseconds)

This is why the Rust Core focuses on the computationally expensive parts (block allocation, scheduling algorithms) while leaving data-heavy parts (token arrays) and complex domain logic (grammar) in Python.