#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use rustc_hash::FxHashMap;
use prost::Message;
use bytes::BytesMut;

// Include the generated protobuf code
pub mod kv_cache {
    include!(concat!(env!("OUT_DIR"), "/kv_cache.rs"));
}

use kv_cache::{Request, Response, request::RequestType, response::ResponseType};
use kv_cache::{AllocateResponse, StatsResponse, BlockIdsResponse, VoidResponse};

const SOCKET_PATH: &str = "ipc:///tmp/kv_cache.sock";

#[repr(C, align(64))]
struct KVCacheBlock {
    id: u32,
    ref_count: u32,
    hash: Option<Box<[u8]>>, // 16 bytes (fat pointer)
    _padding: [u8; 40],
}

impl KVCacheBlock {
    #[tracing::instrument]
    fn new(id: u32) -> Self {
        Self {
            id,
            ref_count: 0,
            hash: None,
            _padding: [0; 40],
        }
    }
}

/// Free block queue using Vec (stack-based for O(1) operations)
struct FreeKVCacheBlockQueue {
    free_blocks: Vec<u32>,
}

impl FreeKVCacheBlockQueue {
    #[tracing::instrument]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            free_blocks: Vec::with_capacity(capacity),
        }
    }

    #[tracing::instrument(skip(self))]
    fn push_front(&mut self, block_id: u32) {
        // Use Vec as a stack (push/pop from end for O(1))
        self.free_blocks.push(block_id);
    }

    #[tracing::instrument(skip(self))]
    fn pop_front(&mut self) -> Option<u32> {
        self.free_blocks.pop()
    }
    
    #[tracing::instrument(skip(self))]
    fn pop_n(&mut self, n: usize) -> Option<Vec<u32>> {
        // Efficiently pop n blocks at once
        let len = self.free_blocks.len();
        if n > len {
            return None;  // Return None instead of panic
        }
        
        // Split off the last n elements (O(1) operation)
        let remaining = len - n;
        let popped = self.free_blocks.split_off(remaining);
        Some(popped)
    }

    #[tracing::instrument(skip(self))]
    fn push_back_batch(&mut self, blocks: Vec<u32>) {
        // For batch operations, we can extend efficiently
        self.free_blocks.extend(blocks);
    }

    #[tracing::instrument(skip(self))]
    fn len(&self) -> usize {
        self.free_blocks.len()
    }

    #[tracing::instrument(skip(self))]
    fn get_all(&self) -> Vec<u32> {
        self.free_blocks.clone()
    }
}

struct BlockPool {
    blocks: Vec<KVCacheBlock>,
    free_list: FreeKVCacheBlockQueue,
    request_id_to_blocks: FxHashMap<String, Vec<u32>>,
    cached_blocks: FxHashMap<Box<[u8]>, u32>,
}

impl BlockPool {
    #[tracing::instrument]
    fn new(num_blocks: u32) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks as usize);
        for i in 0..num_blocks {
            blocks.push(KVCacheBlock::new(i));
        }
        
        let mut free_list = FreeKVCacheBlockQueue::with_capacity(num_blocks as usize);
        // Initialize free list with all blocks
        for i in (0..num_blocks).rev() {
            free_list.push_front(i);
        }
        
        Self {
            blocks,
            free_list,
            request_id_to_blocks: FxHashMap::default(),
            cached_blocks: FxHashMap::default(),
        }
    }

    #[tracing::instrument(skip(self))]
    fn allocate(&mut self, request_id: String, num_blocks: u32) -> Result<Vec<u32>, String> {
        // Pre-check: Ensure we have enough blocks (like Python)
        if num_blocks as usize > self.free_list.len() {
            return Err(format!("Not enough free blocks. Requested: {}, Available: {}", 
                             num_blocks, self.free_list.len()));
        }
        
        // Fast path: pop all blocks at once using split_off (O(1) operation!)
        let allocated = self.free_list.pop_n(num_blocks as usize)
            .ok_or_else(|| format!("Failed to allocate {} blocks", num_blocks))?;
        
        // Update ref counts with prefetching for better cache performance
        // Original implementation (without prefetching):
        for &block_id in &allocated {
            self.blocks[block_id as usize].ref_count = 1;
        }
        
        // Store the allocation
        self.request_id_to_blocks.insert(request_id, allocated.clone());
        Ok(allocated)
    }

    #[tracing::instrument(skip(self))]
    fn free(&mut self, request_id: &str) -> Result<(), String> {
        match self.request_id_to_blocks.remove(request_id) {
            Some(blocks) => {
                for block_id in blocks {
                    let block = &mut self.blocks[block_id as usize];
                    if block.ref_count > 0 {
                        block.ref_count -= 1;
                        if block.ref_count == 0 {
                            // Remove from cache if present
                            if let Some(ref hash) = block.hash {
                                self.cached_blocks.remove(hash);
                            }
                            block.hash = None;
                            self.free_list.push_front(block_id);
                        }
                    }
                }
                Ok(())
            }
            None => Err(format!("Request ID '{}' not found", request_id))
        }
    }

    #[tracing::instrument(skip(self))]
    fn free_blocks_by_id(&mut self, block_ids: Vec<u32>) -> Result<(), String> {
        // Original implementation (without prefetching):
        for block_id in block_ids {
            if block_id >= self.blocks.len() as u32 {
                return Err(format!("Invalid block ID: {}", block_id));
            }
            
            let block = &mut self.blocks[block_id as usize];
            if block.ref_count > 0 {
                block.ref_count -= 1;
                if block.ref_count == 0 {
                    // Remove from cache if present
                    if let Some(ref hash) = block.hash {
                        self.cached_blocks.remove(hash);
                    }
                    block.hash = None;
                    self.free_list.push_front(block_id);
                }
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    fn touch(&mut self, block_ids: Vec<u32>) -> Result<(), String> {
        for block_id in block_ids {
            if block_id >= self.blocks.len() as u32 {
                return Err(format!("Invalid block ID: {}", block_id));
            }
            self.blocks[block_id as usize].ref_count += 1;
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    fn get_cached_block(&self, hash: &str) -> Result<Option<u32>, String> {
        let hash_bytes = hash.as_bytes();
        Ok(self.cached_blocks.get(hash_bytes).copied())
    }

    #[tracing::instrument(skip(self))]
    fn cache_full_blocks(&mut self, hashes: Vec<String>, block_ids: Vec<u32>) -> Result<(), String> {
        if hashes.len() != block_ids.len() {
            return Err("Hashes and block_ids must have the same length".to_string());
        }

        // Decode all hashes
        let mut decoded_hashes = Vec::with_capacity(hashes.len());
        for hash in &hashes {
            let decoded = hash.as_bytes().to_vec().into_boxed_slice();
            decoded_hashes.push(decoded);
        }

        // Update blocks and cache
        for (hash, block_id) in decoded_hashes.into_iter().zip(block_ids) {
            if block_id >= self.blocks.len() as u32 {
                return Err(format!("Invalid block ID: {}", block_id));
            }
            
            self.blocks[block_id as usize].hash = Some(hash.clone());
            self.cached_blocks.insert(hash, block_id);
        }
        
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    fn get_stats(&self) -> (u32, u32, u32) {
        let num_free = self.free_list.len() as u32;
        let num_cached = self.cached_blocks.len() as u32;
        let num_total = self.blocks.len() as u32;
        (num_total, num_free, num_cached)
    }

    #[tracing::instrument(skip(self))]
    fn get_all_free_blocks(&self) -> Vec<u32> {
        self.free_list.get_all()
    }

    #[tracing::instrument(skip(self))]
    fn reset_prefix_cache(&mut self) {
        self.cached_blocks.clear();
        // Clear hashes from all blocks
        for block in &mut self.blocks {
            block.hash = None;
        }
    }
}

#[tracing::instrument(skip(pool, request_bytes))]
fn handle_protobuf_request(pool: &mut BlockPool, request_bytes: &[u8]) -> Result<Vec<u8>, String> {
    // Decode the request
    let request = Request::decode(request_bytes)
        .map_err(|e| format!("Failed to decode request: {}", e))?;
    
    let request_type = request.request_type
        .ok_or_else(|| "Request type not specified".to_string())?;
    
    let mut response = Response {
        success: false,
        error: String::new(),
        response_type: None,
    };
    
    match request_type {
        RequestType::Ping(_) => {
            response.success = true;
            response.response_type = Some(ResponseType::Void(VoidResponse {}));
        }
        
        RequestType::Shutdown(_) => {
            response.success = true;
            response.response_type = Some(ResponseType::Void(VoidResponse {}));
        }
        
        RequestType::Allocate(req) => {
            match pool.allocate(req.request_id, req.num_blocks) {
                Ok(block_ids) => {
                    response.success = true;
                    response.response_type = Some(ResponseType::Allocate(AllocateResponse {
                        block_ids,
                    }));
                }
                Err(e) => {
                    response.error = e;
                }
            }
        }
        
        RequestType::Free(req) => {
            match pool.free(&req.request_id) {
                Ok(()) => {
                    response.success = true;
                    response.response_type = Some(ResponseType::Void(VoidResponse {}));
                }
                Err(e) => {
                    response.error = e;
                }
            }
        }
        
        RequestType::FreeBlocksById(req) => {
            match pool.free_blocks_by_id(req.block_ids) {
                Ok(()) => {
                    response.success = true;
                    response.response_type = Some(ResponseType::Void(VoidResponse {}));
                }
                Err(e) => {
                    response.error = e;
                }
            }
        }
        
        RequestType::Touch(req) => {
            match pool.touch(req.block_ids) {
                Ok(()) => {
                    response.success = true;
                    response.response_type = Some(ResponseType::Void(VoidResponse {}));
                }
                Err(e) => {
                    response.error = e;
                }
            }
        }
        
        RequestType::GetCachedBlock(req) => {
            match pool.get_cached_block(&req.hash) {
                Ok(Some(block_id)) => {
                    response.success = true;
                    response.response_type = Some(ResponseType::BlockIds(BlockIdsResponse {
                        block_ids: vec![block_id],
                    }));
                }
                Ok(None) => {
                    response.success = true;
                    response.response_type = Some(ResponseType::BlockIds(BlockIdsResponse {
                        block_ids: vec![],
                    }));
                }
                Err(e) => {
                    response.error = e;
                }
            }
        }
        
        RequestType::CacheFullBlocks(req) => {
            match pool.cache_full_blocks(req.hashes, req.block_ids) {
                Ok(()) => {
                    response.success = true;
                    response.response_type = Some(ResponseType::Void(VoidResponse {}));
                }
                Err(e) => {
                    response.error = e;
                }
            }
        }
        
        RequestType::GetStats(_) => {
            let (num_total, num_free, num_cached) = pool.get_stats();
            response.success = true;
            response.response_type = Some(ResponseType::Stats(StatsResponse {
                num_total_blocks: num_total,
                num_free_blocks: num_free,
                num_cached_blocks: num_cached,
            }));
        }
        
        RequestType::GetAllFreeBlocks(_) => {
            let free_blocks = pool.get_all_free_blocks();
            response.success = true;
            response.response_type = Some(ResponseType::BlockIds(BlockIdsResponse {
                block_ids: free_blocks,
            }));
        }
        
        RequestType::ResetPrefixCache(_) => {
            pool.reset_prefix_cache();
            response.success = true;
            response.response_type = Some(ResponseType::Void(VoidResponse {}));
        }
    }
    
    // Encode the response (4KB buffer should be sufficient for most responses)
    let mut buf = BytesMut::with_capacity(4 * 1024);
    response.encode(&mut buf)
        .map_err(|e| format!("Failed to encode response: {}", e))?;
    
    Ok(buf.to_vec())
}

#[tracing::instrument]
fn main() {
    // Check if we should enable Chrome tracing for profiling
    let enable_profiling = std::env::var("ENABLE_PROFILING").is_ok();
    
    let _guard = if enable_profiling {
        // Set up Chrome tracing for profiling
        // Save trace in the current working directory
        let trace_path = std::env::current_dir()
            .unwrap()
            .join("trace.json");
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(trace_path.clone())
            .build();
        
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry()
            .with(chrome_layer)
            .init();
        
        println!("Profiling enabled - trace will be saved to {:?}", trace_path);
        Some(guard)
    } else {
        // Regular tracing subscriber for normal operation
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
        None
    };
    
    // Setup signal handler for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    // Use signal-hook for graceful shutdown
    signal_hook::flag::register(signal_hook::consts::SIGINT, r.clone())
        .expect("Error setting SIGINT handler");
    signal_hook::flag::register(signal_hook::consts::SIGTERM, r)
        .expect("Error setting SIGTERM handler");
    
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let num_blocks = if args.len() > 1 {
        args[1].parse::<u32>().unwrap_or(100000)
    } else {
        100000
    };
    
    println!("Starting KV cache server with {} blocks", num_blocks);
    
    // Initialize the block pool
    let mut pool = BlockPool::new(num_blocks);
    
    // Setup ZeroMQ
    let context = zmq::Context::new();
    let responder = context.socket(zmq::REP).unwrap();
    responder.bind(SOCKET_PATH).expect("Failed to bind socket");
    
    // Set receive timeout to allow periodic checking of shutdown signal
    responder.set_rcvtimeo(100).expect("Failed to set receive timeout");
    
    println!("Server listening on {}", SOCKET_PATH);
    
    while running.load(Ordering::SeqCst) {
        let msg = match responder.recv_bytes(0) {
            Ok(m) => m,
            Err(zmq::Error::EAGAIN) => continue,  // Timeout, check if still running
            Err(e) => {
                eprintln!("Error receiving message: {}", e);
                continue;
            }
        };
        
        // Check for shutdown in request
        if let Ok(request) = Request::decode(&msg[..]) {
            if let Some(RequestType::Shutdown(_)) = request.request_type {
                println!("Received shutdown request");
                running.store(false, Ordering::SeqCst);
                
                let response = Response {
                    success: true,
                    error: String::new(),
                    response_type: Some(ResponseType::Void(VoidResponse {})),
                };
                
                let mut buf = BytesMut::with_capacity(64);
                response.encode(&mut buf).unwrap();
                responder.send(&buf[..], 0).unwrap();
                break;
            }
        }
        
        // Handle the request
        match handle_protobuf_request(&mut pool, &msg) {
            Ok(response) => {
                responder.send(&response, 0).expect("Failed to send response");
            }
            Err(e) => {
                eprintln!("Error handling request: {}", e);
                let error_response = Response {
                    success: false,
                    error: e,
                    response_type: None,
                };
                
                let mut buf = BytesMut::with_capacity(256);
                error_response.encode(&mut buf).unwrap();
                responder.send(&buf[..], 0).expect("Failed to send error response");
            }
        }
    }
    
    println!("Server shutting down");
    
    // Ensure tracing is flushed before exit
    if let Some(guard) = _guard {
        drop(guard);
        println!("Profiling trace saved to ./trace.json");
    }
}