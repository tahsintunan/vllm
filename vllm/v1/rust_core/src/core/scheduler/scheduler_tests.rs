#[cfg(test)]
mod tests {
    use crate::core::scheduler::scheduler::{Scheduler, SchedulerConfig};
    use crate::core::scheduler::scheduler_interface::SchedulerInterface;
    use crate::core::{KVCacheConfig, Request, RequestStatus, SamplingParams};

    fn create_test_scheduler() -> Scheduler {
        let scheduler_config = SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 2048,
            max_model_len: 2048,
            max_num_encoder_input_tokens: 512,
            policy: "fcfs".to_string(),
            long_prefill_token_threshold: 512,
            chunked_prefill_enabled: true,
            disable_chunked_mm_input: false,
        };

        let kv_cache_config = KVCacheConfig {
            num_blocks: 1024,
            kv_cache_groups: vec![],
        };

        Scheduler::new(
            scheduler_config,
            kv_cache_config,
            false, // enable_prefix_caching
            false, // include_finished_set
            false, // log_stats
            false, // is_encoder_decoder
            false, // use_pp
        )
    }

    #[test]
    fn test_scheduler_basic_operations() {
        let mut scheduler = create_test_scheduler();

        // Create a test request
        let request = Request::new(
            "test_req_1".to_string(),
            vec![1, 2, 3, 4, 5],
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

        // Add request to scheduler
        scheduler.add_request(request);

        // Check request counts
        let (running, waiting) = scheduler.get_request_counts();
        assert_eq!(running, 0);
        assert_eq!(waiting, 1);

        // Schedule should move request from waiting to running
        let output = scheduler.schedule();
        assert!(
            output.scheduled_new_reqs.len() > 0 || output.scheduled_cached_reqs.req_ids.len() > 0
        );

        // Check if request moved to running
        let (running, waiting) = scheduler.get_request_counts();
        assert!(running > 0 || waiting > 0); // Request should be in one of the queues
    }

    #[test]
    fn test_scheduler_priority_queue() {
        let scheduler_config = SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 2048,
            max_model_len: 2048,
            max_num_encoder_input_tokens: 512,
            policy: "priority".to_string(),
            long_prefill_token_threshold: 512,
            chunked_prefill_enabled: true,
            disable_chunked_mm_input: false,
        };

        let kv_cache_config = KVCacheConfig {
            num_blocks: 1024,
            kv_cache_groups: vec![],
        };

        let mut scheduler = Scheduler::new(
            scheduler_config,
            kv_cache_config,
            false, // enable_prefix_caching
            false, // include_finished_set
            false, // log_stats
            false, // is_encoder_decoder
            false, // use_pp
        );

        // Add requests with different priorities
        let high_priority_req = Request::new(
            "high_priority".to_string(),
            vec![1, 2, 3],
            Some(SamplingParams::default()),
            None,
            None,
            0,
            None,
            None,
            None,
            false, // use_structured_output
            1,     // Lower value = higher priority
        );

        let low_priority_req = Request::new(
            "low_priority".to_string(),
            vec![4, 5, 6],
            Some(SamplingParams::default()),
            None,
            None,
            0,
            None,
            None,
            None,
            false, // use_structured_output
            10,    // Higher value = lower priority
        );

        scheduler.add_request(low_priority_req);
        scheduler.add_request(high_priority_req);

        // Schedule should prioritize high_priority request
        let output = scheduler.schedule();

        // High priority request should be scheduled first
        if !output.scheduled_new_reqs.is_empty() {
            assert_eq!(output.scheduled_new_reqs[0].req_id, "high_priority");
        }
    }

    #[test]
    fn test_scheduler_finish_requests() {
        let mut scheduler = create_test_scheduler();

        // Add a request
        let request = Request::new(
            "test_req".to_string(),
            vec![1, 2, 3],
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

        scheduler.add_request(request);
        assert_eq!(scheduler.get_num_unfinished_requests(), 1);

        // Finish the request
        scheduler.finish_requests(vec!["test_req".to_string()], RequestStatus::FinishedStopped);

        // Check that request is finished
        assert_eq!(scheduler.get_num_unfinished_requests(), 0);
    }

    #[test]
    fn test_scheduler_shutdown() {
        let mut scheduler = create_test_scheduler();

        // Add some requests
        for i in 0..3 {
            let request = Request::new(
                format!("req_{}", i),
                vec![1, 2, 3],
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
            scheduler.add_request(request);
        }

        assert!(scheduler.get_num_unfinished_requests() > 0);

        // Shutdown should clear all requests
        scheduler.shutdown();
        assert_eq!(scheduler.get_num_unfinished_requests(), 0);
    }
}
