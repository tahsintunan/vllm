use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::rc::Rc;

use crate::core::types::request::Request;

use super::RequestQueue;

/// A first-come-first-served queue
pub struct FcfsRequestQueue {
    queue: VecDeque<Rc<RefCell<Request>>>,
}

impl FcfsRequestQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl Default for FcfsRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue for FcfsRequestQueue {
    fn add_request(&mut self, request: Rc<RefCell<Request>>) {
        self.queue.push_back(request);
    }

    fn pop_request(&mut self) -> Option<Rc<RefCell<Request>>> {
        self.queue.pop_front()
    }

    fn peek_request(&self) -> Option<Rc<RefCell<Request>>> {
        self.queue.front().cloned()
    }

    fn prepend_request(&mut self, request: Rc<RefCell<Request>>) {
        self.queue.push_front(request);
    }

    fn prepend_requests(&mut self, requests: &[Rc<RefCell<Request>>]) {
        // Prepend in reverse order to maintain the original order
        for request in requests.iter().rev() {
            self.queue.push_front(request.clone());
        }
    }

    fn remove_request(&mut self, request: Rc<RefCell<Request>>) -> bool {
        let original_len = self.queue.len();
        // Use pointer comparison
        self.queue.retain(|req| !Rc::ptr_eq(req, &request));
        self.queue.len() < original_len
    }

    fn remove_requests(&mut self, requests: &[Rc<RefCell<Request>>]) {
        // Use pointer comparison
        let request_set: HashSet<*const RefCell<Request>> =
            requests.iter().map(|r| Rc::as_ptr(r)).collect();
        self.queue
            .retain(|req| !request_set.contains(&Rc::as_ptr(req)));
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn iter(&self) -> Vec<Rc<RefCell<Request>>> {
        self.queue.iter().cloned().collect()
    }

    fn iter_reversed(&self) -> Vec<Rc<RefCell<Request>>> {
        self.queue.iter().rev().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scheduler::request_queue::test_utils::create_test_request;

    #[test]
    fn test_fcfs_queue() {
        let mut queue = FcfsRequestQueue::new();
        assert!(queue.is_empty());

        let req1 = create_test_request("req1", 5, 1.0);
        let req2 = create_test_request("req2", 1, 2.0);
        let req3 = create_test_request("req3", 3, 3.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());

        assert_eq!(queue.len(), 3);

        // FCFS should return in order of addition
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req2");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3");
        assert!(queue.pop_request().is_none());
    }

    #[test]
    fn test_fcfs_removal() {
        let mut queue = FcfsRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);
        let req3 = create_test_request("req3", 3, 3.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());

        assert_eq!(queue.len(), 3);
        assert!(queue.remove_request(req2.clone()));
        assert_eq!(queue.len(), 2);
        assert!(!queue.remove_request(req2)); // Already removed

        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3");
    }

    #[test]
    fn test_fcfs_prepend() {
        let mut queue = FcfsRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);
        let req3 = create_test_request("req3", 3, 3.0);

        queue.add_request(req3.clone());
        queue.prepend_requests(&[req1.clone(), req2.clone()]);

        // Should be: req1, req2, req3
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req2");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3");
    }

    #[test]
    fn test_fcfs_peek() {
        let mut queue = FcfsRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());

        // Peek multiple times - should always return the same
        for _ in 0..3 {
            assert_eq!(
                queue.peek_request().unwrap().borrow().request_id,
                "req1"
            );
            assert_eq!(queue.len(), 2); // Length unchanged
        }

        // Pop should remove what peek showed
        assert_eq!(
            queue.pop_request().unwrap().borrow().request_id,
            "req1"
        );
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_fcfs_iter() {
        let mut queue = FcfsRequestQueue::new();

        let req1 = create_test_request("req1", 3, 3.0);
        let req2 = create_test_request("req2", 1, 1.0);
        let req3 = create_test_request("req3", 2, 2.0);

        // Add in specific order
        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());

        // Test iter (should maintain insertion order)
        let iter: Vec<_> = queue
            .iter()
            .into_iter()
            .map(|r| r.borrow().request_id.clone())
            .collect();
        assert_eq!(iter, vec!["req1", "req2", "req3"]);

        let rev: Vec<_> = queue
            .iter_reversed()
            .into_iter()
            .map(|r| r.borrow().request_id.clone())
            .collect();
        assert_eq!(rev, vec!["req3", "req2", "req1"]);
    }

    #[test]
    fn test_fcfs_empty_operations() {
        let mut queue = FcfsRequestQueue::new();

        assert!(queue.pop_request().is_none());
        assert!(queue.peek_request().is_none());
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());

        let req = create_test_request("test", 1, 1.0);
        assert!(!queue.remove_request(req));

        assert!(queue.iter().is_empty());
        assert!(queue.iter_reversed().is_empty());
    }

    #[test]
    fn test_fcfs_mixed_operations() {
        let mut queue = FcfsRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);
        let req3 = create_test_request("req3", 3, 3.0);
        let req4 = create_test_request("req4", 4, 4.0);

        // Complex sequence of operations
        queue.add_request(req1.clone());
        queue.prepend_request(req2.clone());
        queue.add_request(req3.clone());
        assert_eq!(queue.len(), 3);

        assert!(queue.remove_request(req1.clone()));
        assert_eq!(queue.len(), 2);

        queue.prepend_requests(&[req4.clone(), req1.clone()]);
        assert_eq!(queue.len(), 4);

        // Should be: req4, req1, req2, req3
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req4");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req2");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3");
    }
}