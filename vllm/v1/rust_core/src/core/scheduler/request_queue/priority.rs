use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::rc::Rc;

use crate::core::types::request::Request;

use super::RequestQueue;

#[derive(Clone)]
struct PriorityRequestItem {
    request: Rc<RefCell<Request>>,
    /// Lower value = higher priority (processed first)
    priority: i32,
    /// Earlier time = higher priority when priorities are equal
    arrival_time: f64,
}

impl PriorityRequestItem {
    fn new(request: Rc<RefCell<Request>>) -> Self {
        let priority = request.borrow().priority;
        let arrival_time = request.borrow().arrival_time;
        Self {
            priority,
            arrival_time,
            request,
        }
    }
}

// Implement ordering for the priority queue.
// Note: BinaryHeap is a max-heap by default, so we need to reverse the comparison
// to get min-heap behavior (smaller priority values processed first)
impl PartialEq for PriorityRequestItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.arrival_time == other.arrival_time
    }
}

impl Eq for PriorityRequestItem {}

impl Ord for PriorityRequestItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (reversed for min-heap)
        match other.priority.cmp(&self.priority) {
            Ordering::Equal => {
                // If priorities are equal, compare by arrival time (reversed for min-heap)
                other
                    .arrival_time
                    .partial_cmp(&self.arrival_time)
                    .unwrap_or(Ordering::Equal)
            }
            other_order => other_order,
        }
    }
}

impl PartialOrd for PriorityRequestItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A priority queue that processes requests with smaller priority values first
/// If priorities are equal, earlier arrival_time is processed first
pub struct PriorityRequestQueue {
    heap: BinaryHeap<PriorityRequestItem>,
}

impl PriorityRequestQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }
}

impl Default for PriorityRequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue for PriorityRequestQueue {
    fn add_request(&mut self, request: Rc<RefCell<Request>>) {
        self.heap.push(PriorityRequestItem::new(request));
    }

    fn pop_request(&mut self) -> Option<Rc<RefCell<Request>>> {
        self.heap.pop().map(|item| item.request)
    }

    fn peek_request(&self) -> Option<Rc<RefCell<Request>>> {
        self.heap.peek().map(|item| item.request.clone())
    }

    fn prepend_request(&mut self, request: Rc<RefCell<Request>>) {
        // In a priority queue, there's no concept of prepending
        // Requests are ordered by (priority, arrival_time)
        self.add_request(request);
    }

    fn prepend_requests(&mut self, requests: &[Rc<RefCell<Request>>]) {
        // In a priority queue, there's no concept of prepending
        // Requests are ordered by (priority, arrival_time)
        for request in requests {
            self.add_request(request.clone());
        }
    }

    fn remove_request(&mut self, request: Rc<RefCell<Request>>) -> bool {
        let original_len = self.heap.len();

        // Collect all items except the one to remove
        let items: Vec<_> = self
            .heap
            .drain()
            .filter(|item| !Rc::ptr_eq(&item.request, &request))
            .collect();

        // Rebuild the heap
        self.heap = BinaryHeap::from(items);

        self.heap.len() < original_len
    }

    fn remove_requests(&mut self, requests: &[Rc<RefCell<Request>>]) {
        let request_set: HashSet<*const RefCell<Request>> =
            requests.iter().map(|r| Rc::as_ptr(r)).collect();

        // Collect all items except those to remove
        let items: Vec<_> = self
            .heap
            .drain()
            .filter(|item| !request_set.contains(&Rc::as_ptr(&item.request)))
            .collect();

        // Rebuild the heap
        self.heap = BinaryHeap::from(items);
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn iter(&self) -> Vec<Rc<RefCell<Request>>> {
        // Create a sorted vector from the heap
        // Note: Ord is reversed for min-heap, so we need reverse sort to get ascending priority
        let mut items: Vec<_> = self.heap.iter().cloned().collect();
        items.sort_by(|a, b| b.cmp(a)); // Reverse sort
        items.into_iter().map(|item| item.request).collect()
    }

    fn iter_reversed(&self) -> Vec<Rc<RefCell<Request>>> {
        // Create a sorted vector from the heap
        // Normal sort gives descending priority order (what we want for reversed)
        let mut items: Vec<_> = self.heap.iter().cloned().collect();
        items.sort();
        items.into_iter().map(|item| item.request).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scheduler::request_queue::test_utils::create_test_request;

    #[test]
    fn test_priority_queue() {
        let mut queue = PriorityRequestQueue::new();
        assert!(queue.is_empty());

        let req1 = create_test_request("req1", 5, 1.0);
        let req2 = create_test_request("req2", 1, 2.0);
        let req3 = create_test_request("req3", 3, 3.0);
        let req4 = create_test_request("req4", 1, 1.5); // Same priority as req2, earlier arrival

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());
        queue.add_request(req4.clone());

        assert_eq!(queue.len(), 4);

        // Should return in priority order, with arrival time as tiebreaker
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req4"); // priority 1, arrival 1.5
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req2"); // priority 1, arrival 2.0
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3"); // priority 3
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1"); // priority 5
        assert!(queue.pop_request().is_none());
    }

    #[test]
    fn test_priority_removal() {
        let mut queue = PriorityRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);
        let req3 = create_test_request("req3", 3, 3.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());

        // Remove middle priority
        assert!(queue.remove_request(req2.clone()));
        assert_eq!(queue.len(), 2);

        // Should still maintain priority order
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3");

        // Can't remove already removed
        assert!(!queue.remove_request(req2));
    }

    #[test]
    fn test_priority_same_priority() {
        let mut queue = PriorityRequestQueue::new();

        // All same priority, different arrival times
        let req1 = create_test_request("req1", 5, 3.0);
        let req2 = create_test_request("req2", 5, 1.0);
        let req3 = create_test_request("req3", 5, 2.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());

        // Should come out in arrival time order
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req2"); // earliest
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1"); // latest
    }

    #[test]
    fn test_priority_edge_values() {
        let mut queue = PriorityRequestQueue::new();

        // Test with extreme priority values
        let req_min = create_test_request("min", i32::MIN, 1.0);
        let req_max = create_test_request("max", i32::MAX, 2.0);
        let req_zero = create_test_request("zero", 0, 3.0);
        let req_neg = create_test_request("neg", -100, 4.0);

        queue.add_request(req_max.clone());
        queue.add_request(req_zero.clone());
        queue.add_request(req_min.clone());
        queue.add_request(req_neg.clone());

        // Should come out in priority order (lower = higher priority)
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "min");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "neg");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "zero");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "max");
    }

    #[test]
    fn test_priority_peek() {
        let mut queue = PriorityRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());

        // Peek multiple times - should always return the same
        for _ in 0..3 {
            assert_eq!(queue.peek_request().unwrap().borrow().request_id, "req1");
            assert_eq!(queue.len(), 2);
        }

        // Pop should remove what peek showed
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_priority_iter() {
        let mut queue = PriorityRequestQueue::new();

        let req1 = create_test_request("req1", 3, 3.0);
        let req2 = create_test_request("req2", 1, 1.0);
        let req3 = create_test_request("req3", 2, 2.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());

        // Test iter - should return items in ascending priority order (matching Python)
        let items: Vec<_> = queue.iter().into_iter().collect();
        assert_eq!(items.len(), 3);

        // Verify items are sorted by priority in ascending order
        let priorities: Vec<_> = items.iter().map(|r| r.borrow().priority).collect();
        assert_eq!(priorities, vec![1, 2, 3]); // Ascending order like Python

        // Test reversed iterator - should return in descending priority order
        let rev_items: Vec<_> = queue.iter_reversed().into_iter().collect();
        assert_eq!(rev_items.len(), 3);

        let rev_priorities: Vec<_> = rev_items.iter().map(|r| r.borrow().priority).collect();
        assert_eq!(rev_priorities, vec![3, 2, 1]); // Descending order
    }

    #[test]
    fn test_priority_prepend_behaves_as_add() {
        let mut queue = PriorityRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);

        // Priority queue prepend just adds (doesn't actually prepend)
        queue.add_request(req2.clone());
        queue.prepend_request(req1.clone());

        // Should still be in priority order
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req2");
    }

    #[test]
    fn test_priority_remove_multiple() {
        let mut queue = PriorityRequestQueue::new();

        let req1 = create_test_request("req1", 1, 1.0);
        let req2 = create_test_request("req2", 2, 2.0);
        let req3 = create_test_request("req3", 3, 3.0);
        let req4 = create_test_request("req4", 4, 4.0);

        queue.add_request(req1.clone());
        queue.add_request(req2.clone());
        queue.add_request(req3.clone());
        queue.add_request(req4.clone());

        // Remove multiple at once
        queue.remove_requests(&[req2, req4]);

        assert_eq!(queue.len(), 2);

        // Check remaining are correct
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req1");
        assert_eq!(queue.pop_request().unwrap().borrow().request_id, "req3");
    }
}
