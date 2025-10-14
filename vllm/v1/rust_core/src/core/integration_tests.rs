/// Integration tests for vLLM Rust Core components
/// These tests verify the interaction between different modules

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::core::{
        create_request_queue,
        kv_cache::attention_managers::FullAttentionSpec,
        FcfsRequestQueue,
        KVCacheConfig,
        KVCacheGroup,
        // KV Cache components
        KVCacheManager,
        NewRequestData,

        PriorityRequestQueue,
        // Request and Queue components
        Request,
        RequestQueue,
        RequestStatus,
        // Scheduler Output components
        SchedulerOutput,
        SchedulingPolicy,
    };

    fn create_test_config() -> KVCacheConfig {
        KVCacheConfig {
            num_blocks: 100,
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        }
    }

    // =========================================================================
    // Integration Test 1: RequestQueue -> SchedulerOutput flow
    // Tests that requests from queue can be properly converted to scheduler output
    // =========================================================================

    #[test]
    fn test_request_queue_to_scheduler_output_integration() {
        // Create a FCFS queue and add requests
        let mut queue = FcfsRequestQueue::new();

        // Add multiple requests
        for i in 0..5 {
            let request = Request::new_simple(
                format!("req-{}", i),
                10 + i * 5, // Different prompt lengths
            );
            queue.add_request(Rc::new(RefCell::new(request)));
        }

        // Create scheduler output
        let mut output = SchedulerOutput::new();

        // Process requests from queue to scheduler output
        while let Some(request_rc) = queue.pop_request() {
            let request = request_rc.borrow();

            // Simulate block allocation (in real scheduler this would involve KVCacheManager)
            let block_ids = vec![vec![1u32, 2], vec![3, 4]];

            // Add to scheduler output as new request
            let new_req_data = NewRequestData::from_request(&*request, block_ids);
            let num_tokens = request.prompt_token_ids.len();
            output.add_new_request(new_req_data, num_tokens);
        }

        // Verify scheduler output
        assert_eq!(output.scheduled_new_reqs.len(), 5);
        assert_eq!(output.num_scheduled_reqs(), 5);
        assert!(output.total_num_scheduled_tokens > 0);

        // Verify request IDs match
        for i in 0..5 {
            assert_eq!(output.scheduled_new_reqs[i].req_id, format!("req-{}", i));
        }
    }

    // =========================================================================
    // Integration Test 2: Request -> KVCacheManager -> SchedulerOutput
    // Tests the complete flow from request through cache allocation to output
    // =========================================================================

    #[test]
    fn test_request_kvcache_scheduler_integration() {
        // Create KV cache manager
        let config = create_test_config();
        let mut kv_manager = KVCacheManager::new(
            config, 1024,  // max_model_len
            false, // enable_caching
            false, // use_eagle
            false, // log_stats
            false, // enable_kv_cache_events
            1,     // dcp_world_size
        );

        // Create a request
        let request = Request::new_simple("test-req".to_string(), 32);

        // Allocate KV cache blocks
        let allocated = kv_manager.allocate_slots(&request, 32, 0, None, 0, false, 0);
        assert!(allocated.is_some());

        // Get computed blocks
        let (_blocks, num_cached) = kv_manager.get_computed_blocks(&request);
        assert_eq!(num_cached, 0); // No caching since it's disabled

        // Create scheduler output with the allocated blocks
        let mut output = SchedulerOutput::new();

        // Get block IDs from manager
        let block_ids = kv_manager.get_block_ids(&request.request_id);
        let new_req_data = NewRequestData::from_request(&request, block_ids);
        output.add_new_request(new_req_data, request.prompt_token_ids.len());

        // Verify integration
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "test-req");
        assert_eq!(output.scheduled_new_reqs[0].num_computed_tokens, 0);

        // Clean up
        kv_manager.free(&request);
    }

    // =========================================================================
    // Integration Test 3: Priority Queue -> KVCacheManager resource contention
    // Tests that priority queue ordering works correctly with cache allocation
    // =========================================================================

    #[test]
    fn test_priority_queue_kvcache_resource_contention() {
        // Create KV cache manager with limited blocks
        let config = KVCacheConfig {
            num_blocks: 10, // Only 10 blocks total
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        };
        let mut kv_manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        // Create priority queue
        let mut queue = PriorityRequestQueue::new();

        // Add requests with different priorities
        // NOTE: Lower priority values are processed first (like nice values in Unix)
        let high_priority_req = Request {
            priority: 1, // Lowest value = highest priority
            ..Request::new_simple("high-priority".to_string(), 32)
        };
        let low_priority_req = Request {
            priority: 10, // Higher value = lower priority
            ..Request::new_simple("low-priority".to_string(), 32)
        };
        let medium_priority_req = Request {
            priority: 5,
            ..Request::new_simple("medium-priority".to_string(), 32)
        };

        // Add in random order
        queue.add_request(Rc::new(RefCell::new(low_priority_req)));
        queue.add_request(Rc::new(RefCell::new(high_priority_req)));
        queue.add_request(Rc::new(RefCell::new(medium_priority_req)));

        // Create scheduler output
        let mut output = SchedulerOutput::new();
        let mut allocated_requests = Vec::new();

        // Process requests in priority order with cache allocation
        while let Some(request_rc) = queue.pop_request() {
            let request = request_rc.borrow();

            // Try to allocate cache
            let allocation = kv_manager.allocate_slots(
                &*request,
                request.prompt_token_ids.len(),
                0,
                None,
                0,
                false,
                0,
            );
            if allocation.is_some() {
                // Successfully allocated, add to output
                let block_ids = kv_manager.get_block_ids(&request.request_id);
                let new_req_data = NewRequestData::from_request(&*request, block_ids);
                output.add_new_request(new_req_data, request.prompt_token_ids.len());
                allocated_requests.push(request.request_id.clone());
            } else {
                // Not enough blocks, would need preemption in real scheduler
                break;
            }
        }

        // Verify high priority was scheduled first
        assert!(!allocated_requests.is_empty());
        assert_eq!(allocated_requests[0], "high-priority");

        // With 10 blocks, each request needs 2 blocks (32/16)
        // So we can fit at most 5 requests (but we only have 3)
        assert!(allocated_requests.len() <= 3);
    }

    // =========================================================================
    // Integration Test 4: Request state transitions with SchedulerOutput
    // Tests request lifecycle from WAITING -> RUNNING with outputs
    // =========================================================================

    #[test]
    fn test_request_lifecycle_with_scheduler_output() {
        // Create components
        let config = create_test_config();
        let mut kv_manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);

        // Create request in waiting state
        let mut request = Request::new_simple("lifecycle-req".to_string(), 32);
        assert_eq!(request.status, RequestStatus::Waiting);

        // First scheduling: WAITING -> RUNNING (new request)
        request.status = RequestStatus::Running;
        let allocated = kv_manager.allocate_slots(&request, 32, 0, None, 0, false, 0);
        assert!(allocated.is_some());

        let mut output1 = SchedulerOutput::new();
        let block_ids = kv_manager.get_block_ids(&request.request_id);
        let new_req_data = NewRequestData::from_request(&request, block_ids);
        output1.add_new_request(new_req_data, request.prompt_token_ids.len());

        assert_eq!(output1.scheduled_new_reqs.len(), 1);
        assert_eq!(output1.scheduled_cached_reqs.num_reqs(), 0);

        // Second scheduling: RUNNING (cached request, generation step)
        let mut output2 = SchedulerOutput::new();
        request.num_computed_tokens = 32; // Prompt processed
        request.output_token_ids.push(100); // Generated one token

        output2.add_cached_request(
            request.request_id.clone(),
            false,     // not resumed from preemption
            vec![100], // new token
            None,      // no new blocks needed yet
            request.num_computed_tokens + 1,
            1, // one token scheduled
        );

        assert_eq!(output2.scheduled_new_reqs.len(), 0);
        assert_eq!(output2.scheduled_cached_reqs.num_reqs(), 1);

        // Third scheduling: Mark as finished (using FinishedLengthCapped)
        request.status = RequestStatus::FinishedLengthCapped;
        let mut output3 = SchedulerOutput::new();
        output3.mark_finished(request.request_id.clone());

        assert!(output3.finished_req_ids.contains(&request.request_id));

        // Clean up
        kv_manager.free(&request);
    }

    // =========================================================================
    // Integration Test 5: Preemption scenario with RequestQueue and SchedulerOutput
    // Tests handling of preempted requests going back to queue
    // =========================================================================

    #[test]
    fn test_preemption_flow_integration() {
        // Create a queue
        let mut queue = FcfsRequestQueue::new();

        // Simulate a running request that gets preempted
        let mut running_request = Request::new_simple("running".to_string(), 64);
        running_request.status = RequestStatus::Running;
        running_request.num_computed_tokens = 32; // Partially processed

        // Create output showing the preemption
        let mut preemption_output = SchedulerOutput::new();

        // Add as cached request being resumed from preemption
        preemption_output.add_cached_request(
            running_request.request_id.clone(),
            true,                            // resumed from preemption!
            vec![],                          // no new tokens
            Some(vec![vec![5u32, 6, 7, 8]]), // new blocks after preemption
            running_request.num_computed_tokens,
            0, // no tokens scheduled (just resuming)
        );

        // Verify preemption flag
        assert!(
            preemption_output
                .scheduled_cached_reqs
                .resumed_from_preemption[0]
        );

        // In real scheduler, preempted request would go back to queue
        running_request.status = RequestStatus::Preempted;
        queue.prepend_requests(&[Rc::new(RefCell::new(running_request.clone()))]);

        // Verify it's back at the front of queue
        let resumed = queue.pop_request().unwrap();
        assert_eq!(resumed.borrow().request_id, "running");
        assert_eq!(resumed.borrow().num_computed_tokens, 32); // Still remembers progress
    }

    // =========================================================================
    // Integration Test 6: Queue factory function with different policies
    // Tests the create_request_queue factory function
    // =========================================================================

    #[test]
    fn test_request_queue_factory_integration() {
        // Create FCFS queue via factory
        let mut fcfs_queue = create_request_queue(SchedulingPolicy::Fcfs);

        // Create Priority queue via factory
        let mut priority_queue = create_request_queue(SchedulingPolicy::Priority);

        // Add same requests to both
        // NOTE: Lower priority values are processed first
        let req1 = Request {
            priority: 10,        // Lower priority (higher value)
            arrival_time: 100.0, // Arrived later
            ..Request::new_simple("req1".to_string(), 16)
        };
        let req2 = Request {
            priority: 5,        // Higher priority (lower value)
            arrival_time: 50.0, // Arrived earlier
            ..Request::new_simple("req2".to_string(), 16)
        };

        // For FCFS: Add in order of arrival (req2 arrived first with arrival_time=50)
        // In real system, requests would be added as they arrive
        fcfs_queue.add_request(Rc::new(RefCell::new(req2.clone())));
        fcfs_queue.add_request(Rc::new(RefCell::new(req1.clone())));

        // For Priority: Order doesn't matter, heap will sort
        priority_queue.add_request(Rc::new(RefCell::new(req1.clone())));
        priority_queue.add_request(Rc::new(RefCell::new(req2.clone())));

        // FCFS should return based on insertion order (req2 was added first)
        let fcfs_first = fcfs_queue.pop_request().unwrap();
        assert_eq!(fcfs_first.borrow().request_id, "req2");

        // Priority should return based on priority value (req2 has priority=5 < req1's priority=10)
        let priority_first = priority_queue.pop_request().unwrap();
        assert_eq!(priority_first.borrow().request_id, "req2");
    }

    // =========================================================================
    // Integration Test 7: SchedulerOutput with multiple request types
    // Tests mixing new and cached requests in a single output
    // =========================================================================

    #[test]
    fn test_scheduler_output_mixed_requests() {
        let mut output = SchedulerOutput::new();

        // Add new requests
        let new_req1 = Request::new_simple("new-1".to_string(), 16);
        let new_req2 = Request::new_simple("new-2".to_string(), 32);

        output.add_new_request(
            NewRequestData::from_request(&new_req1, vec![vec![1u32, 2]]),
            new_req1.prompt_token_ids.len(),
        );
        output.add_new_request(
            NewRequestData::from_request(&new_req2, vec![vec![3u32, 4, 5]]),
            new_req2.prompt_token_ids.len(),
        );

        // Add cached requests
        output.add_cached_request("cached-1".to_string(), false, vec![100], None, 50, 1);
        output.add_cached_request(
            "cached-2".to_string(),
            true, // resumed from preemption
            vec![],
            Some(vec![vec![6u32, 7]]),
            30,
            0,
        );

        // Add finished requests
        output.mark_finished("finished-1".to_string());
        output.mark_finished("finished-2".to_string());

        // Verify counts
        assert_eq!(output.scheduled_new_reqs.len(), 2);
        assert_eq!(output.scheduled_cached_reqs.num_reqs(), 2);
        assert_eq!(output.finished_req_ids.len(), 2);
        assert_eq!(output.num_scheduled_reqs(), 4);
        assert_eq!(output.total_num_scheduled_tokens, 16 + 32 + 1 + 0);
    }

    // =========================================================================
    // Integration Test 8: Request with block hashing for prefix caching
    // Tests that block hashes are properly handled
    // =========================================================================

    #[test]
    fn test_request_with_block_hashing() {
        // Create KV cache manager with caching enabled
        let config = create_test_config();
        let mut kv_manager = KVCacheManager::new(
            config, 1024, true, // enable caching
            false, true, // log stats
            false, 1,
        );

        // Create request with block hashes
        let mut request = Request::new_simple("hash-req".to_string(), 32);
        request.block_hashes = vec![[1u8; 32], [2u8; 32]]; // Two blocks worth

        // Allocate and cache
        let allocated = kv_manager.allocate_slots(&request, 32, 0, None, 0, false, 0);
        assert!(allocated.is_some());

        // Cache the blocks
        kv_manager.cache_blocks(&request, 32);

        // Create scheduler output
        let mut output = SchedulerOutput::new();
        let block_ids = kv_manager.get_block_ids(&request.request_id);
        let new_req_data = NewRequestData::from_request(&request, block_ids);
        output.add_new_request(new_req_data, request.prompt_token_ids.len());

        // Verify request was added
        assert_eq!(output.scheduled_new_reqs.len(), 1);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "hash-req");

        // Clean up
        kv_manager.free(&request);

        // Create second request with same hashes
        let mut request2 = Request::new_simple("hash-req-2".to_string(), 32);
        request2.block_hashes = vec![[1u8; 32], [2u8; 32]];

        // Should get cache hit
        let (computed_blocks, hit_tokens) = kv_manager.get_computed_blocks(&request2);
        assert_eq!(hit_tokens, 16); // One block cached (see manager tests for why)
        assert_eq!(computed_blocks.blocks[0].len(), 1);
    }

    // =========================================================================
    // Integration Test 9: Complete scheduling batch
    // Tests a realistic scheduling scenario with multiple components
    // =========================================================================

    #[test]
    fn test_complete_scheduling_batch() {
        // Setup components
        let config = KVCacheConfig {
            num_blocks: 50,
            kv_cache_groups: vec![KVCacheGroup {
                kv_cache_spec: Box::new(FullAttentionSpec {
                    block_size: 16,
                    sliding_window: None,
                }),
            }],
        };
        let mut kv_manager = KVCacheManager::new(config, 1024, false, false, false, false, 1);
        let mut queue = PriorityRequestQueue::new();

        // Add multiple requests with different characteristics
        for i in 0..5 {
            let mut request = Request::new_simple(format!("req-{}", i), 16 + i * 8);
            request.priority = (5 - i) as i32; // Decreasing priority
            queue.add_request(Rc::new(RefCell::new(request)));
        }

        // Scheduling cycle
        let mut output = SchedulerOutput::new();
        let mut running_requests = Vec::new();

        // Schedule as many as can fit in KV cache
        while let Some(request_rc) = queue.pop_request() {
            let request = request_rc.borrow();

            let allocation = kv_manager.allocate_slots(
                &*request,
                request.prompt_token_ids.len(),
                0,
                None,
                0,
                false,
                0,
            );

            if allocation.is_some() {
                let block_ids = kv_manager.get_block_ids(&request.request_id);
                let new_req_data = NewRequestData::from_request(&*request, block_ids);
                output.add_new_request(new_req_data, request.prompt_token_ids.len());
                running_requests.push(request_rc.clone());
            } else {
                // No more space, put back in queue
                drop(request);
                queue.add_request(request_rc);
                break;
            }
        }

        // Verify scheduling results
        assert!(output.scheduled_new_reqs.len() > 0);
        assert_eq!(output.scheduled_new_reqs.len(), running_requests.len());

        // Simulate one generation step for running requests
        let mut next_output = SchedulerOutput::new();
        for request_rc in &running_requests {
            let mut request = request_rc.borrow_mut();
            let token = 100 + request.priority; // Mock token
            request.output_token_ids.push(token);
            request.num_computed_tokens = request.prompt_token_ids.len();

            next_output.add_cached_request(
                request.request_id.clone(),
                false,
                vec![100 + request.priority],
                None,
                request.num_computed_tokens + 1,
                1,
            );
        }

        assert_eq!(
            next_output.scheduled_cached_reqs.num_reqs(),
            running_requests.len()
        );
        assert_eq!(next_output.scheduled_new_reqs.len(), 0);

        // Clean up
        for request_rc in &running_requests {
            kv_manager.free(&*request_rc.borrow());
        }
    }

    // =========================================================================
    // Integration Test 10: RequestQueue removal and prepend operations
    // Tests queue manipulation operations work correctly with SchedulerOutput
    // =========================================================================

    #[test]
    fn test_queue_operations_with_scheduler_output() {
        let mut queue = FcfsRequestQueue::new();
        let mut output = SchedulerOutput::new();

        // Add multiple requests
        let req1 = Request::new_simple("req-1".to_string(), 16);
        let req2 = Request::new_simple("req-2".to_string(), 32);
        let req3 = Request::new_simple("req-3".to_string(), 48);

        let req1_rc = Rc::new(RefCell::new(req1));
        let req2_rc = Rc::new(RefCell::new(req2));
        let req3_rc = Rc::new(RefCell::new(req3));

        queue.add_request(req1_rc.clone());
        queue.add_request(req2_rc.clone());
        queue.add_request(req3_rc.clone());

        // Remove middle request
        let removed = queue.remove_request(req2_rc);
        assert!(removed, "Failed to remove req2 from queue");
        assert_eq!(queue.len(), 2);

        // Process remaining requests to output
        while let Some(request_rc) = queue.pop_request() {
            let request = request_rc.borrow();
            let new_req_data = NewRequestData::from_request(&*request, vec![vec![1u32]]);
            output.add_new_request(new_req_data, request.prompt_token_ids.len());
        }

        // Should have only req-1 and req-3
        assert_eq!(output.scheduled_new_reqs.len(), 2);
        assert_eq!(output.scheduled_new_reqs[0].req_id, "req-1");
        assert_eq!(output.scheduled_new_reqs[1].req_id, "req-3");

        // Test prepend operation
        let urgent_req = Request::new_simple("urgent".to_string(), 16);
        queue.prepend_requests(&[Rc::new(RefCell::new(urgent_req))]);

        // Urgent request should be first
        let first = queue.pop_request().unwrap();
        assert_eq!(first.borrow().request_id, "urgent");
    }
}
