use std::cell::RefCell;
use std::rc::Rc;

use crate::core::types::request::Request;

mod fcfs;
mod priority;

pub use fcfs::FcfsRequestQueue;
pub use priority::PriorityRequestQueue;

/// Enum for scheduling policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    Fcfs,
    Priority,
}

/// Trait for request queues
pub trait RequestQueue {
    /// Add a request to the queue according to the policy
    fn add_request(&mut self, request: Rc<RefCell<Request>>);

    /// Pop a request from the queue according to the policy
    fn pop_request(&mut self) -> Option<Rc<RefCell<Request>>>;

    /// Peek at the request at the front of the queue without removing it
    fn peek_request(&self) -> Option<Rc<RefCell<Request>>>;

    /// Prepend a request to the front of the queue
    fn prepend_request(&mut self, request: Rc<RefCell<Request>>);

    /// Prepend all requests from another queue to the front of this queue
    fn prepend_requests(&mut self, requests: &[Rc<RefCell<Request>>]);

    /// Remove a specific request from the queue
    fn remove_request(&mut self, request: Rc<RefCell<Request>>) -> bool;

    /// Remove multiple specific requests from the queue
    fn remove_requests(&mut self, requests: &[Rc<RefCell<Request>>]);

    /// Check if queue has any requests
    fn is_empty(&self) -> bool;

    /// Get number of requests in queue
    fn len(&self) -> usize;

    /// Get all requests in order (for iteration)
    fn iter(&self) -> Vec<Rc<RefCell<Request>>>;

    /// Get all requests in reverse order
    fn iter_reversed(&self) -> Vec<Rc<RefCell<Request>>>;
}

/// Create a request queue based on scheduling policy
pub fn create_request_queue(policy: SchedulingPolicy) -> Box<dyn RequestQueue> {
    match policy {
        SchedulingPolicy::Priority => Box::new(PriorityRequestQueue::new()),
        SchedulingPolicy::Fcfs => Box::new(FcfsRequestQueue::new()),
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::SamplingParams;

    pub fn create_test_request(
        id: &str,
        priority: i32,
        arrival_time: f64,
    ) -> Rc<RefCell<Request>> {
        let request = Request::new(
            id.to_string(),
            vec![1, 2, 3],
            Some(SamplingParams::default()),
            None,
            None,
            0,
            Some(arrival_time),
            None,
            None,
            false, // use_structured_output
            priority,
        );
        Rc::new(RefCell::new(request))
    }
}