# request.rs is not a part of Core in Python, rather a dependency
- Keep Request in Rust core for performance
- At the FFI/IPC boundary, convert Python Request → Rust Request
- This minimizes cross-language calls during scheduling
- The Request in Rust would be a "scheduling view" of the full Python Request (Rust Request is a subset focused only on what the scheduler/block manager **NEEDS** to make decisions. The actual token data, sampling parameters, etc. stay in Python where they're used for the actual inference.)
- Trim fields that are not needed in Core.

# i32, i64
Double-check if they really nead to be signed integer.

# RequestQueue - Priority Queue Performance Issue
- **Problem**: BinaryHeap remove is O(n) - drains entire heap + filters + rebuilds from scratch
- **Impact**: Removing 1 request from 1000 = touching all 1000 items + allocations
- **Solution**: Switch to BTreeSet for O(log n) removes while keeping O(log n) pop/push
- **Alternatives**: BTreeSet, RBTree, priority_queue::PriorityQueue, skiplist::SkipList

# Add Capacity Hints (request_queue)
- VecDeque::new() → with_capacity() in RequestQueue
- Vec::new() → with_capacity() in scheduler (use max_num_running_reqs)
- HashMap::new() → with_capacity() where sizes are predictable

# Unused
- kv_cache_utils
- scheduler_interface
- Scheduler -> request_queue, scheduler_output, encoder_cache_manager, request