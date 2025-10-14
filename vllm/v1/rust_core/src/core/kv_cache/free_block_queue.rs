use std::cell::RefCell;
use std::rc::Rc;

pub type BlockHash = [u8; 32];
pub type BlockRef = Rc<RefCell<KVCacheBlock>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockHashWithGroupId {
    pub hash: BlockHash,
    pub group_id: u32,
}

/// `KVCacheBlock` represents a single memory block on the GPU that can store KV pairs.
#[derive(Debug)]
pub struct KVCacheBlock {
    /// Unique block ID ranging from `(0 - num_gpu_blocks-1)`
    pub block_id: u32,
    pub ref_cnt: u32,
    /// The hash key `(block_hash + group_id)` of the block, only available when the block is full and cached.
    pub hash_key: Option<BlockHashWithGroupId>,
    /// Null blocks should never be cached.
    pub is_null: bool,

    // Used to construct a doubly linked list for free blocks.
    // These two attributes should only be manipulated by `FreeKVCacheBlockQueue`.
    pub prev_free_block: Option<BlockRef>,
    pub next_free_block: Option<BlockRef>,
}

impl KVCacheBlock {
    pub fn new(block_id: u32) -> Self {
        Self {
            block_id,
            ref_cnt: 0,
            hash_key: None,
            is_null: false,
            prev_free_block: None,
            next_free_block: None,
        }
    }
}

/// A queue of unused blocks ready for allocation. It's implemented as a doubly-linked list with:
/// - Fake head and tail nodes (sentinels) to simplify operations
/// - Methods to pop blocks from the front, remove from middle, or append to the end
///
/// The queue is ordered by block ID in the beginning. When a block is allocated
/// and then freed, it will be appended back with the eviction order:
/// 1. The least recent used block is at the front (LRU).
/// 2. If two blocks have the same last accessed time (allocated by the
///    same sequence), the one with more hash tokens (the tail of a block
///    chain) is at the front.
///    Note that we maintain this order by reversing the block order when free
///    blocks of a request. This reversal happens in `SingleTypeKVCacheManager`.
pub struct FreeKVCacheBlockQueue {
    num_free_blocks: usize,
    fake_head: BlockRef,
    fake_tail: BlockRef,
}

impl FreeKVCacheBlockQueue {
    pub fn new(blocks: Vec<BlockRef>) -> Self {
        let num_free_blocks = blocks.len();

        // Create sentinel nodes (assuming we'll never have more than 4 billion blocks!)
        let fake_head = Rc::new(RefCell::new(KVCacheBlock::new(u32::MAX)));
        let fake_tail = Rc::new(RefCell::new(KVCacheBlock::new(u32::MAX - 1)));

        if !blocks.is_empty() {
            // Connect fake_head to first block
            fake_head.borrow_mut().next_free_block = Some(blocks[0].clone());
            blocks[0].borrow_mut().prev_free_block = Some(fake_head.clone());

            // Connect blocks to each other
            for i in 0..blocks.len() {
                if i > 0 {
                    blocks[i].borrow_mut().prev_free_block = Some(blocks[i - 1].clone());
                }
                if i < blocks.len() - 1 {
                    blocks[i].borrow_mut().next_free_block = Some(blocks[i + 1].clone());
                }
            }

            // Connect last block to fake_tail
            let last_idx = blocks.len() - 1;
            blocks[last_idx].borrow_mut().next_free_block = Some(fake_tail.clone());
            fake_tail.borrow_mut().prev_free_block = Some(blocks[last_idx].clone());
        } else {
            // Empty list - connect sentinels directly
            fake_head.borrow_mut().next_free_block = Some(fake_tail.clone());
            fake_tail.borrow_mut().prev_free_block = Some(fake_head.clone());
        }

        Self {
            num_free_blocks,
            fake_head,
            fake_tail,
        }
    }

    /// Pop the first free block and reduce `num_free_blocks` by 1.
    pub fn popleft(&mut self) -> Option<BlockRef> {
        if self.num_free_blocks == 0 {
            return None;
        }

        let first_block = self
            .fake_head
            .borrow()
            .next_free_block
            .clone()
            .expect("fake_head should have next when num_free_blocks > 0");
        let second_block = first_block
            .borrow()
            .next_free_block
            .clone()
            .expect("first block should have next (at least fake_tail)");

        // Reconnect fake_head to second block
        self.fake_head.borrow_mut().next_free_block = Some(second_block.clone());
        second_block.borrow_mut().prev_free_block = Some(self.fake_head.clone());

        // Clear the pointers in the removed block
        first_block.borrow_mut().prev_free_block = None;
        first_block.borrow_mut().next_free_block = None;

        self.num_free_blocks -= 1;
        Some(first_block)
    }

    /// Pop the first `n` free blocks and reduce `num_free_blocks` by `n`
    pub fn popleft_n(&mut self, n: usize) -> Vec<BlockRef> {
        if n == 0 || self.num_free_blocks < n {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(n);
        let mut current = self
            .fake_head
            .borrow()
            .next_free_block
            .clone()
            .expect("fake_head.next must exist when queue has blocks");

        for _ in 0..n {
            let next = current
                .borrow()
                .next_free_block
                .clone()
                .expect("block.next must exist in valid doubly-linked list");

            // Clear pointers in removed block
            current.borrow_mut().prev_free_block = None;
            current.borrow_mut().next_free_block = None;

            result.push(current.clone());
            current = next;
        }

        // Connect fake_head to the new first block
        self.fake_head.borrow_mut().next_free_block = Some(current.clone());
        current.borrow_mut().prev_free_block = Some(self.fake_head.clone());

        self.num_free_blocks -= n;
        result
    }

    /// Remove a particular block from the free list and reduce `num_free_blocks` by 1.
    /// Returns `true` if the block was successfully removed, `false` if the block was not found in the list.
    pub fn remove(&mut self, block: &BlockRef) -> bool {
        let prev = match &block.borrow().prev_free_block {
            Some(p) => p.clone(),
            None => return false,
        };

        let next = match &block.borrow().next_free_block {
            Some(n) => n.clone(),
            None => return false,
        };

        // Unlink the block
        prev.borrow_mut().next_free_block = Some(next.clone());
        next.borrow_mut().prev_free_block = Some(prev);

        // Clear the block's pointers
        block.borrow_mut().prev_free_block = None;
        block.borrow_mut().next_free_block = None;

        self.num_free_blocks -= 1;
        true
    }

    /// Put a block back into the free list and increase `num_free_blocks` by 1.
    pub fn append(&mut self, block: BlockRef) {
        // Get the current last block (before fake_tail)
        let last = self
            .fake_tail
            .borrow()
            .prev_free_block
            .clone()
            .expect("fake_tail.prev must always exist");

        // Insert block between last and fake_tail
        block.borrow_mut().prev_free_block = Some(last.clone());
        block.borrow_mut().next_free_block = Some(self.fake_tail.clone());

        last.borrow_mut().next_free_block = Some(block.clone());
        self.fake_tail.borrow_mut().prev_free_block = Some(block);

        self.num_free_blocks += 1;
    }

    /// Put `n` blocks back into the free list and increase `num_free_blocks` by `n`.
    pub fn append_n(&mut self, blocks: &[BlockRef]) {
        if blocks.is_empty() {
            return;
        }

        // Get current last block
        let mut last = self
            .fake_tail
            .borrow()
            .prev_free_block
            .clone()
            .expect("fake_tail.prev must always exist");

        // Link all new blocks
        for block in blocks {
            block.borrow_mut().prev_free_block = Some(last.clone());
            last.borrow_mut().next_free_block = Some(block.clone());
            last = block.clone();
        }

        // Connect last new block to fake_tail
        last.borrow_mut().next_free_block = Some(self.fake_tail.clone());
        self.fake_tail.borrow_mut().prev_free_block = Some(last);

        self.num_free_blocks += blocks.len();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.num_free_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test blocks
    fn create_blocks(n: u32) -> Vec<BlockRef> {
        (0..n)
            .map(|i| Rc::new(RefCell::new(KVCacheBlock::new(i))))
            .collect()
    }

    // ===== Queue Initialization Tests =====

    #[test]
    fn test_empty_queue_initialization() {
        let queue = FreeKVCacheBlockQueue::new(vec![]);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_single_block_queue() {
        let blocks = create_blocks(1);
        let mut queue = FreeKVCacheBlockQueue::new(blocks.clone());

        assert_eq!(queue.len(), 1);

        let popped = queue.popleft().unwrap();
        assert_eq!(popped.borrow().block_id, 0);
        assert_eq!(queue.len(), 0);

        // Should return None when empty
        assert!(queue.popleft().is_none());
    }

    #[test]
    fn test_multiple_blocks_initialization() {
        let blocks = create_blocks(10);
        let queue = FreeKVCacheBlockQueue::new(blocks);
        assert_eq!(queue.len(), 10);
    }

    // ===== popleft() Tests =====

    #[test]
    fn test_popleft_from_empty_queue() {
        let mut queue = FreeKVCacheBlockQueue::new(vec![]);
        assert!(queue.popleft().is_none());
    }

    #[test]
    fn test_popleft_clears_pointers() {
        let blocks = create_blocks(3);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let popped = queue.popleft().unwrap();

        // Verify pointers are cleared
        assert!(popped.borrow().prev_free_block.is_none());
        assert!(popped.borrow().next_free_block.is_none());
    }

    #[test]
    fn test_popleft_until_empty() {
        let blocks = create_blocks(3);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        // Pop all blocks
        for expected_id in 0..3 {
            let block = queue.popleft().unwrap();
            assert_eq!(block.borrow().block_id, expected_id);
        }

        assert_eq!(queue.len(), 0);
        assert!(queue.popleft().is_none());
    }

    #[test]
    fn test_popleft_maintains_order() {
        let blocks = create_blocks(5);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        // Blocks should come out in order
        for expected_id in 0..5 {
            let block = queue.popleft().unwrap();
            assert_eq!(block.borrow().block_id, expected_id);
        }
    }

    // ===== popleft_n() Tests =====

    #[test]
    fn test_popleft_n_zero() {
        let blocks = create_blocks(3);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let popped = queue.popleft_n(0);
        assert_eq!(popped.len(), 0);
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_popleft_n_more_than_available() {
        let blocks = create_blocks(3);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let popped = queue.popleft_n(10);
        assert_eq!(popped.len(), 0);
        assert_eq!(queue.len(), 3); // Queue unchanged
    }

    #[test]
    fn test_popleft_n_exact() {
        let blocks = create_blocks(5);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let popped = queue.popleft_n(5);
        assert_eq!(popped.len(), 5);
        assert_eq!(queue.len(), 0);

        // Verify order
        for (i, block) in popped.iter().enumerate() {
            assert_eq!(block.borrow().block_id, i as u32);
        }
    }

    #[test]
    fn test_popleft_n_partial() {
        let blocks = create_blocks(5);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let popped = queue.popleft_n(3);
        assert_eq!(popped.len(), 3);
        assert_eq!(queue.len(), 2);

        // Check returned blocks
        for (i, block) in popped.iter().enumerate() {
            assert_eq!(block.borrow().block_id, i as u32);
        }

        // Check remaining blocks
        let remaining = queue.popleft_n(2);
        assert_eq!(remaining[0].borrow().block_id, 3);
        assert_eq!(remaining[1].borrow().block_id, 4);
    }

    #[test]
    fn test_popleft_n_clears_all_pointers() {
        let blocks = create_blocks(3);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let popped = queue.popleft_n(2);

        for block in popped {
            assert!(block.borrow().prev_free_block.is_none());
            assert!(block.borrow().next_free_block.is_none());
        }
    }

    // ===== remove() Tests =====

    #[test]
    fn test_remove_from_empty_queue() {
        let mut queue = FreeKVCacheBlockQueue::new(vec![]);
        let block = Rc::new(RefCell::new(KVCacheBlock::new(99)));

        assert!(!queue.remove(&block));
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_remove_block_not_in_queue() {
        let blocks = create_blocks(3);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let external_block = Rc::new(RefCell::new(KVCacheBlock::new(99)));
        assert!(!queue.remove(&external_block));
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_remove_first_block() {
        let blocks = create_blocks(3);
        let first_block = blocks[0].clone();
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        assert!(queue.remove(&first_block));
        assert_eq!(queue.len(), 2);

        // Verify remaining order
        let next = queue.popleft().unwrap();
        assert_eq!(next.borrow().block_id, 1);
    }

    #[test]
    fn test_remove_middle_block() {
        let blocks = create_blocks(5);
        let middle_block = blocks[2].clone();
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        assert!(queue.remove(&middle_block));
        assert_eq!(queue.len(), 4);

        // Verify order is maintained (0, 1, 3, 4)
        let remaining = queue.popleft_n(4);
        assert_eq!(remaining[0].borrow().block_id, 0);
        assert_eq!(remaining[1].borrow().block_id, 1);
        assert_eq!(remaining[2].borrow().block_id, 3);
        assert_eq!(remaining[3].borrow().block_id, 4);
    }

    #[test]
    fn test_remove_last_block() {
        let blocks = create_blocks(3);
        let last_block = blocks[2].clone();
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        assert!(queue.remove(&last_block));
        assert_eq!(queue.len(), 2);

        let remaining = queue.popleft_n(2);
        assert_eq!(remaining[0].borrow().block_id, 0);
        assert_eq!(remaining[1].borrow().block_id, 1);
    }

    #[test]
    fn test_remove_only_block() {
        let blocks = create_blocks(1);
        let block = blocks[0].clone();
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        assert!(queue.remove(&block));
        assert_eq!(queue.len(), 0);
        assert!(queue.popleft().is_none());
    }

    #[test]
    fn test_remove_same_block_twice() {
        let blocks = create_blocks(3);
        let block = blocks[1].clone();
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        assert!(queue.remove(&block));
        assert_eq!(queue.len(), 2);

        // Second remove should fail
        assert!(!queue.remove(&block));
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_remove_clears_pointers() {
        let blocks = create_blocks(3);
        let block = blocks[1].clone();
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        assert!(queue.remove(&block));

        // Verify pointers are cleared
        assert!(block.borrow().prev_free_block.is_none());
        assert!(block.borrow().next_free_block.is_none());
    }

    // ===== append() Tests =====

    #[test]
    fn test_append_to_empty_queue() {
        let mut queue = FreeKVCacheBlockQueue::new(vec![]);
        let block = Rc::new(RefCell::new(KVCacheBlock::new(42)));

        queue.append(block.clone());
        assert_eq!(queue.len(), 1);

        let popped = queue.popleft().unwrap();
        assert_eq!(popped.borrow().block_id, 42);
    }

    #[test]
    fn test_append_to_non_empty_queue() {
        let blocks = create_blocks(2);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let new_block = Rc::new(RefCell::new(KVCacheBlock::new(99)));
        queue.append(new_block.clone());

        assert_eq!(queue.len(), 3);

        // Verify order: original blocks come first
        let all = queue.popleft_n(3);
        assert_eq!(all[0].borrow().block_id, 0);
        assert_eq!(all[1].borrow().block_id, 1);
        assert_eq!(all[2].borrow().block_id, 99);
    }

    #[test]
    fn test_append_previously_removed_block() {
        let blocks = create_blocks(3);
        let block = blocks[1].clone();
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        assert!(queue.remove(&block));
        assert_eq!(queue.len(), 2);

        queue.append(block.clone());
        assert_eq!(queue.len(), 3);

        // Block should be at the end now
        let all = queue.popleft_n(3);
        assert_eq!(all[0].borrow().block_id, 0);
        assert_eq!(all[1].borrow().block_id, 2);
        assert_eq!(all[2].borrow().block_id, 1); // Previously removed, now at end
    }

    // ===== append_n() Tests =====

    #[test]
    fn test_append_n_empty_slice() {
        let blocks = create_blocks(2);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        queue.append_n(&[]);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_append_n_single_block() {
        let blocks = create_blocks(2);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let new_block = Rc::new(RefCell::new(KVCacheBlock::new(99)));
        queue.append_n(&[new_block.clone()]);

        assert_eq!(queue.len(), 3);

        let all = queue.popleft_n(3);
        assert_eq!(all[2].borrow().block_id, 99);
    }

    #[test]
    fn test_append_n_to_empty_queue() {
        let mut queue = FreeKVCacheBlockQueue::new(vec![]);

        let new_blocks = create_blocks(3);
        queue.append_n(&new_blocks);

        assert_eq!(queue.len(), 3);

        for i in 0..3 {
            let block = queue.popleft().unwrap();
            assert_eq!(block.borrow().block_id, i);
        }
    }

    #[test]
    fn test_append_n_multiple_blocks() {
        let blocks = create_blocks(2);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        let new_blocks: Vec<BlockRef> = (10..13)
            .map(|i| Rc::new(RefCell::new(KVCacheBlock::new(i))))
            .collect();

        queue.append_n(&new_blocks);
        assert_eq!(queue.len(), 5);

        // Verify order
        let all = queue.popleft_n(5);
        assert_eq!(all[0].borrow().block_id, 0);
        assert_eq!(all[1].borrow().block_id, 1);
        assert_eq!(all[2].borrow().block_id, 10);
        assert_eq!(all[3].borrow().block_id, 11);
        assert_eq!(all[4].borrow().block_id, 12);
    }

    // ===== LRU Ordering Tests =====

    #[test]
    fn test_lru_ordering_with_append() {
        let blocks = create_blocks(3);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        // Pop some blocks
        let b0 = queue.popleft().unwrap();
        let b1 = queue.popleft().unwrap();

        // Append them back in reverse order
        queue.append(b1.clone());
        queue.append(b0.clone());

        // Queue should now be: 2, 1, 0
        let all = queue.popleft_n(3);
        assert_eq!(all[0].borrow().block_id, 2);
        assert_eq!(all[1].borrow().block_id, 1);
        assert_eq!(all[2].borrow().block_id, 0);
    }

    #[test]
    fn test_fifo_ordering_maintained() {
        let blocks = create_blocks(5);
        let mut queue = FreeKVCacheBlockQueue::new(blocks);

        // Remove some from middle
        let b0 = queue.popleft().unwrap();
        let b1 = queue.popleft().unwrap();

        // Add them back at end
        queue.append_n(&[b0, b1]);

        // Should be: 2, 3, 4, 0, 1
        let all = queue.popleft_n(5);
        assert_eq!(all[0].borrow().block_id, 2);
        assert_eq!(all[1].borrow().block_id, 3);
        assert_eq!(all[2].borrow().block_id, 4);
        assert_eq!(all[3].borrow().block_id, 0);
        assert_eq!(all[4].borrow().block_id, 1);
    }
}
