use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json;
use tokio::sync::{mpsc, oneshot};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info, warn};
use zmq;

// Include the generated protobuf code
pub mod kv_cache {
    include!(concat!(env!("OUT_DIR"), "/kv_cache.rs"));
}

use kv_cache::kv_cache_service_server::{KvCacheService, KvCacheServiceServer};
use kv_cache::*;

// Import the actual KV cache manager
use rust_core::core::kv_cache::kv_cache_coordinator::KVCacheConfig;
use rust_core::core::kv_cache::kv_cache_manager::KVCacheManager;
use rust_core::core::types::request::Request as CoreRequest;

// ZMQ Request/Response structures
#[derive(Serialize, Deserialize, Debug)]
struct ZmqRequest {
    method: String,
    params: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct ZmqResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(flatten)]
    data: serde_json::Value,
}

// Commands to send to the KV cache manager thread
#[derive(Debug)]
enum KvCommand {
    AllocateSlots {
        request: CoreRequest,
        num_computed_tokens: usize,
        num_new_tokens: usize,
        spec_token_ids: Vec<i32>,
        response: oneshot::Sender<Result<Vec<Vec<i32>>, String>>,
    },
    Free {
        request_id: String,
        response: oneshot::Sender<Result<(), String>>,
    },
    GetUsage {
        response: oneshot::Sender<f32>,
    },
    GetBlocks {
        request_id: String,
        response: oneshot::Sender<Result<Vec<Vec<i32>>, String>>,
    },
    GetBlockIds {
        request_id: String,
        response: oneshot::Sender<Result<Vec<Vec<i32>>, String>>,
    },
    GetComputedBlocks {
        request: CoreRequest,
        response: oneshot::Sender<(Vec<Vec<i32>>, usize)>,
    },
    CacheBlocks {
        request: CoreRequest,
        num_computed_tokens: usize,
        response: oneshot::Sender<Result<(), String>>,
    },
    GetNumFreeBlocks {
        response: oneshot::Sender<usize>,
    },
    ResetPrefixCache {
        response: oneshot::Sender<bool>,
    },
    Shutdown,
}

/// Start ZMQ server in a separate thread
fn start_zmq_server(command_tx: mpsc::Sender<KvCommand>) {
    let zmq_port = std::env::var("ZMQ_PORT").unwrap_or_else(|_| "5555".to_string());

    std::thread::spawn(move || {
        info!("Starting ZMQ server on port {}", zmq_port);

        let context = zmq::Context::new();
        let responder = context.socket(zmq::REP).unwrap();
        responder
            .bind(&format!("tcp://127.0.0.1:{}", zmq_port))
            .expect("Failed to bind ZMQ socket");

        info!("ZMQ server listening on port {}", zmq_port);

        loop {
            // Receive request
            let msg = match responder.recv_msg(0) {
                Ok(msg) => msg,
                Err(e) => {
                    error!("ZMQ receive error: {}", e);
                    continue;
                }
            };

            // Parse JSON request
            let request: ZmqRequest = match serde_json::from_slice(msg.as_ref()) {
                Ok(req) => req,
                Err(e) => {
                    error!("Failed to parse ZMQ request: {}", e);
                    let error_resp = ZmqResponse {
                        success: false,
                        error: Some(format!("Invalid JSON: {}", e)),
                        data: serde_json::Value::Null,
                    };
                    let _ = responder.send(serde_json::to_vec(&error_resp).unwrap(), 0);
                    continue;
                }
            };

            // Handle request based on method
            let response = match request.method.as_str() {
                "initialize" => {
                    // Initialization is handled at startup
                    ZmqResponse {
                        success: true,
                        error: None,
                        data: serde_json::json!({
                            "message": "Already initialized"
                        }),
                    }
                }
                "get_usage" => {
                    let (tx, rx) = std::sync::mpsc::sync_channel(1);
                    let async_tx = command_tx.clone();

                    // Use blocking runtime to send async command
                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async move {
                            let (response_tx, response_rx) = oneshot::channel();
                            let _ = async_tx.send(KvCommand::GetUsage { response: response_tx }).await;
                            if let Ok(usage) = response_rx.await {
                                let _ = tx.send(usage);
                            }
                        });
                    });

                    match rx.recv_timeout(std::time::Duration::from_secs(2)) {
                        Ok(usage) => ZmqResponse {
                            success: true,
                            error: None,
                            data: serde_json::json!({ "usage": usage }),
                        },
                        Err(_) => ZmqResponse {
                            success: false,
                            error: Some("Timeout getting usage".to_string()),
                            data: serde_json::Value::Null,
                        }
                    }
                }
                "reset_prefix_cache" => {
                    let (tx, rx) = std::sync::mpsc::sync_channel(1);
                    let async_tx = command_tx.clone();

                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async move {
                            let (response_tx, response_rx) = oneshot::channel();
                            let _ = async_tx.send(KvCommand::ResetPrefixCache { response: response_tx }).await;
                            if let Ok(success) = response_rx.await {
                                let _ = tx.send(success);
                            }
                        });
                    });

                    match rx.recv_timeout(std::time::Duration::from_secs(2)) {
                        Ok(success) => ZmqResponse {
                            success,
                            error: None,
                            data: serde_json::Value::Null,
                        },
                        Err(_) => ZmqResponse {
                            success: false,
                            error: Some("Timeout resetting prefix cache".to_string()),
                            data: serde_json::Value::Null,
                        }
                    }
                }
                "get_num_free_blocks" => {
                    let (tx, rx) = std::sync::mpsc::sync_channel(1);
                    let async_tx = command_tx.clone();

                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async move {
                            let (response_tx, response_rx) = oneshot::channel();
                            let _ = async_tx.send(KvCommand::GetNumFreeBlocks { response: response_tx }).await;
                            if let Ok(free_blocks) = response_rx.await {
                                let _ = tx.send(free_blocks);
                            }
                        });
                    });

                    match rx.recv_timeout(std::time::Duration::from_secs(2)) {
                        Ok(free_blocks) => ZmqResponse {
                            success: true,
                            error: None,
                            data: serde_json::json!({ "num_free_blocks": free_blocks }),
                        },
                        Err(_) => ZmqResponse {
                            success: false,
                            error: Some("Timeout getting free blocks".to_string()),
                            data: serde_json::Value::Null,
                        }
                    }
                }
                "shutdown" => {
                    info!("Received shutdown request via ZMQ");
                    let async_tx = command_tx.clone();
                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        rt.block_on(async move {
                            let _ = async_tx.send(KvCommand::Shutdown).await;
                        });
                    });

                    ZmqResponse {
                        success: true,
                        error: None,
                        data: serde_json::json!({ "message": "Shutting down" }),
                    }
                }
                // TODO: Implement other methods (allocate_slots, free, etc.)
                _ => ZmqResponse {
                    success: false,
                    error: Some(format!("Method '{}' not implemented", request.method)),
                    data: serde_json::Value::Null,
                }
            };

            // Send response
            let response_json = serde_json::to_vec(&response).unwrap();
            if let Err(e) = responder.send(response_json, 0) {
                error!("Failed to send ZMQ response: {}", e);
            }

            // Exit on shutdown
            if request.method == "shutdown" {
                break;
            }
        }

        info!("ZMQ server shutting down");
    });
}

/// KV Cache gRPC service implementation
pub struct KvCacheServiceImpl {
    command_tx: mpsc::Sender<KvCommand>,
    running: Arc<AtomicBool>,
    start_time: Instant,
    request_count: Arc<AtomicU64>,
}

impl KvCacheServiceImpl {
    fn new(num_blocks: usize, enable_prefix_caching: bool) -> Self {
        // Create channel for communication with KV cache manager thread
        let (command_tx, mut command_rx) = mpsc::channel::<KvCommand>(100);

        // Start ZMQ server for Python client communication
        start_zmq_server(command_tx.clone());

        // Spawn the KV cache manager thread
        std::thread::spawn(move || {
            // Create the actual KV cache manager
            let kv_config = KVCacheConfig {
                num_blocks,
                kv_cache_groups: vec![],
            };

            let mut kv_manager = KVCacheManager::new(
                kv_config,
                32768,                 // max_model_len
                enable_prefix_caching, // enable_caching
                false,                 // use_eagle
                false,                 // log_stats
                false,                 // enable_kv_cache_events
                1,                     // dcp_world_size
            );

            // Track requests by ID for operations that need them
            let mut request_map: std::collections::HashMap<String, CoreRequest> = std::collections::HashMap::new();

            // Run the event loop
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async {
                while let Some(cmd) = command_rx.recv().await {
                    match cmd {
                        KvCommand::AllocateSlots {
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            spec_token_ids,
                            response,
                        } => {
                            // Store request for later operations
                            request_map.insert(request.request_id.clone(), request.clone());

                            // Call kv_manager.allocate_slots with correct signature
                            let blocks_opt = kv_manager.allocate_slots(
                                &request,
                                num_new_tokens,        // num_new_tokens
                                num_computed_tokens,   // num_new_computed_tokens
                                None,                  // new_computed_blocks
                                spec_token_ids.len(),  // num_lookahead_tokens
                                false,                 // delay_cache_blocks
                                0,                     // num_encoder_tokens
                            );

                            // Convert Option<KVCacheBlocks> to Vec<Vec<i32>>
                            let block_ids = if let Some(blocks) = blocks_opt {
                                blocks.get_block_ids()
                                    .into_iter()
                                    .map(|group| group.into_iter().map(|id| id as i32).collect())
                                    .collect()
                            } else {
                                vec![]
                            };

                            let _ = response.send(Ok(block_ids));
                        }
                        KvCommand::Free {
                            request_id,
                            response,
                        } => {
                            // Find and free the request
                            if let Some(request) = request_map.remove(&request_id) {
                                kv_manager.free(&request);
                                let _ = response.send(Ok(()));
                            } else {
                                let _ = response.send(Err(format!("Request {} not found", request_id)));
                            }
                        }
                        KvCommand::GetUsage { response } => {
                            let usage = kv_manager.usage();
                            let _ = response.send(usage);
                        }
                        KvCommand::GetBlocks {
                            request_id,
                            response,
                        } => {
                            // Call kv_manager.get_blocks
                            let blocks = kv_manager.get_blocks(&request_id);
                            let block_ids = blocks.get_block_ids()
                                .into_iter()
                                .map(|group| group.into_iter().map(|id| id as i32).collect())
                                .collect();
                            let _ = response.send(Ok(block_ids));
                        }
                        KvCommand::GetBlockIds {
                            request_id,
                            response,
                        } => {
                            // Call kv_manager.get_block_ids
                            let block_ids = kv_manager.get_block_ids(&request_id)
                                .into_iter()
                                .map(|group| group.into_iter().map(|id| id as i32).collect())
                                .collect();
                            let _ = response.send(Ok(block_ids));
                        }
                        KvCommand::GetComputedBlocks { request, response } => {
                            // Store request if not already there
                            request_map.entry(request.request_id.clone())
                                .or_insert_with(|| request.clone());

                            let (blocks, num_tokens) = kv_manager.get_computed_blocks(&request);
                            let block_ids = blocks.get_block_ids()
                                .into_iter()
                                .map(|group| group.into_iter().map(|id| id as i32).collect())
                                .collect();
                            let _ = response.send((block_ids, num_tokens));
                        }
                        KvCommand::CacheBlocks { request, num_computed_tokens, response } => {
                            // Store request if not already there
                            request_map.entry(request.request_id.clone())
                                .or_insert_with(|| request.clone());

                            kv_manager.cache_blocks(&request, num_computed_tokens);
                            let _ = response.send(Ok(()));
                        }
                        KvCommand::GetNumFreeBlocks { response } => {
                            // Get free blocks from block pool
                            // Note: KVCacheManager doesn't expose this directly,
                            // we can calculate from usage
                            let usage = kv_manager.usage();
                            let total_blocks = num_blocks;
                            let used_blocks = (usage * total_blocks as f32) as usize;
                            let free_blocks = total_blocks - used_blocks;
                            let _ = response.send(free_blocks);
                        }
                        KvCommand::ResetPrefixCache { response } => {
                            let result = kv_manager.reset_prefix_cache();
                            let _ = response.send(result);
                        }
                        KvCommand::Shutdown => {
                            info!("KV cache manager thread shutting down");
                            break;
                        }
                    }
                }
            });

            info!("KV cache manager thread exited");
        });

        Self {
            command_tx,
            running: Arc::new(AtomicBool::new(true)),
            start_time: Instant::now(),
            request_count: Arc::new(AtomicU64::new(0)),
        }
    }

    fn increment_request_count(&self) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }
}

#[tonic::async_trait]
impl KvCacheService for KvCacheServiceImpl {
    async fn initialize(
        &self,
        request: Request<InitializeRequest>,
    ) -> Result<Response<InitializeResponse>, Status> {
        self.increment_request_count();
        info!("Initialize request received");

        let req = request.into_inner();

        // In a real implementation, we would reinitialize the scheduler here
        // For now, we just return success
        let response = InitializeResponse {
            success: true,
            error: String::new(),
            total_blocks: req.config.map_or(0, |c| c.num_blocks),
        };

        Ok(Response::new(response))
    }

    async fn allocate_slots(
        &self,
        request: Request<AllocateSlotsRequest>,
    ) -> Result<Response<AllocateSlotsResponse>, Status> {
        self.increment_request_count();

        let req = request.into_inner();
        let request_info = req
            .request
            .ok_or_else(|| Status::invalid_argument("request field is required"))?;

        // TODO: Convert protobuf Request to our internal Request type
        // and call scheduler.schedule()

        // For now, return a dummy response
        let response = AllocateSlotsResponse {
            success: true,
            error: String::new(),
            allocated_blocks: vec![],
            block_ids: vec![],
        };

        Ok(Response::new(response))
    }

    async fn free_blocks(
        &self,
        request: Request<FreeBlocksRequest>,
    ) -> Result<Response<FreeBlocksResponse>, Status> {
        self.increment_request_count();

        let req = request.into_inner();

        // TODO: Call scheduler to free blocks for the request

        let response = FreeBlocksResponse {
            success: true,
            error: String::new(),
            freed_blocks: 0,
        };

        Ok(Response::new(response))
    }

    async fn get_computed_blocks(
        &self,
        request: Request<GetComputedBlocksRequest>,
    ) -> Result<Response<GetComputedBlocksResponse>, Status> {
        self.increment_request_count();

        let response = GetComputedBlocksResponse {
            success: true,
            error: String::new(),
            computed_blocks: vec![],
            num_computed_tokens: 0,
        };

        Ok(Response::new(response))
    }

    async fn get_num_free_blocks(
        &self,
        _request: Request<GetNumFreeBlocksRequest>,
    ) -> Result<Response<GetNumFreeBlocksResponse>, Status> {
        self.increment_request_count();

        // TODO: Get actual free blocks from KV manager
        // For now, return a placeholder
        let response = GetNumFreeBlocksResponse {
            num_free_blocks: 1000,
        };

        Ok(Response::new(response))
    }

    async fn get_usage(
        &self,
        _request: Request<GetUsageRequest>,
    ) -> Result<Response<GetUsageResponse>, Status> {
        self.increment_request_count();

        let (response_tx, response_rx) = oneshot::channel();

        self.command_tx
            .send(KvCommand::GetUsage {
                response: response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to send command to KV manager"))?;

        let usage = response_rx
            .await
            .map_err(|_| Status::internal("Failed to receive response from KV manager"))?;

        // TODO: Get actual block counts from KV manager
        let response = GetUsageResponse {
            usage,
            used_blocks: 0,  // TODO: Track this
            total_blocks: 0, // TODO: Track this
        };

        Ok(Response::new(response))
    }

    async fn reset_prefix_cache(
        &self,
        _request: Request<ResetPrefixCacheRequest>,
    ) -> Result<Response<ResetPrefixCacheResponse>, Status> {
        self.increment_request_count();

        let (response_tx, response_rx) = oneshot::channel();

        self.command_tx
            .send(KvCommand::ResetPrefixCache {
                response: response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to send command to KV manager"))?;

        let success = response_rx
            .await
            .map_err(|_| Status::internal("Failed to receive response from KV manager"))?;

        let response = ResetPrefixCacheResponse {
            success,
            error: if success {
                String::new()
            } else {
                "Failed to reset prefix cache".to_string()
            },
        };

        Ok(Response::new(response))
    }

    async fn get_blocks(
        &self,
        request: Request<GetBlocksRequest>,
    ) -> Result<Response<GetBlocksResponse>, Status> {
        self.increment_request_count();

        let req = request.into_inner();
        let (response_tx, response_rx) = oneshot::channel();

        self.command_tx
            .send(KvCommand::GetBlocks {
                request_id: req.request_id,
                response: response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to send command to KV manager"))?;

        match response_rx.await {
            Ok(Ok(blocks)) => {
                // Flatten the blocks
                let flattened: Vec<u32> = blocks.into_iter().flatten().map(|b| b as u32).collect();
                let response = GetBlocksResponse {
                    success: true,
                    error: String::new(),
                    block_ids: flattened,
                };
                Ok(Response::new(response))
            }
            Ok(Err(e)) => {
                let response = GetBlocksResponse {
                    success: false,
                    error: e,
                    block_ids: vec![],
                };
                Ok(Response::new(response))
            }
            Err(_) => Err(Status::internal(
                "Failed to receive response from KV manager",
            )),
        }
    }

    async fn get_block_ids(
        &self,
        request: Request<GetBlockIdsRequest>,
    ) -> Result<Response<GetBlockIdsResponse>, Status> {
        self.increment_request_count();

        let req = request.into_inner();
        let (response_tx, response_rx) = oneshot::channel();

        self.command_tx
            .send(KvCommand::GetBlockIds {
                request_id: req.request_id,
                response: response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to send command to KV manager"))?;

        match response_rx.await {
            Ok(Ok(blocks)) => {
                // Flatten the blocks
                let flattened: Vec<u32> = blocks.into_iter().flatten().map(|b| b as u32).collect();
                let response = GetBlockIdsResponse {
                    success: true,
                    error: String::new(),
                    block_ids: flattened,
                };
                Ok(Response::new(response))
            }
            Ok(Err(e)) => {
                let response = GetBlockIdsResponse {
                    success: false,
                    error: e,
                    block_ids: vec![],
                };
                Ok(Response::new(response))
            }
            Err(_) => Err(Status::internal(
                "Failed to receive response from KV manager",
            )),
        }
    }

    async fn cache_blocks(
        &self,
        request: Request<CacheBlocksRequest>,
    ) -> Result<Response<CacheBlocksResponse>, Status> {
        self.increment_request_count();

        // TODO: Implement cache_blocks
        let response = CacheBlocksResponse {
            success: false,
            error: "Not implemented yet".to_string(),
        };

        Ok(Response::new(response))
    }

    async fn take_events(
        &self,
        _request: Request<TakeEventsRequest>,
    ) -> Result<Response<TakeEventsResponse>, Status> {
        self.increment_request_count();

        // TODO: Implement take_events
        let response = TakeEventsResponse { events: vec![] };

        Ok(Response::new(response))
    }

    async fn get_num_common_prefix_blocks(
        &self,
        _request: Request<GetNumCommonPrefixBlocksRequest>,
    ) -> Result<Response<GetNumCommonPrefixBlocksResponse>, Status> {
        self.increment_request_count();

        // TODO: Implement get_num_common_prefix_blocks
        let response = GetNumCommonPrefixBlocksResponse {
            num_common_blocks: 0,
        };

        Ok(Response::new(response))
    }

    async fn get_prefix_cache_stats(
        &self,
        _request: Request<GetPrefixCacheStatsRequest>,
    ) -> Result<Response<GetPrefixCacheStatsResponse>, Status> {
        self.increment_request_count();

        // TODO: Implement get_prefix_cache_stats
        let response = GetPrefixCacheStatsResponse {
            has_stats: false,
            stats: None,
        };

        Ok(Response::new(response))
    }

    async fn shutdown(
        &self,
        request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
        info!("Shutdown request received");

        let req = request.into_inner();
        self.running.store(false, Ordering::SeqCst);

        let response = ShutdownResponse {
            success: true,
            message: if req.graceful {
                "Graceful shutdown initiated".to_string()
            } else {
                "Immediate shutdown initiated".to_string()
            },
        };

        Ok(Response::new(response))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let uptime = self.start_time.elapsed().as_secs();
        let request_count = self.request_count.load(Ordering::Relaxed);

        let response = HealthCheckResponse {
            healthy: self.running.load(Ordering::SeqCst),
            status: format!("Running. Processed {} requests", request_count),
            uptime_seconds: uptime,
        };

        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rust_core=info".parse()?),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let num_blocks = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(10000)
    } else {
        10000
    };

    let addr = std::env::var("GRPC_ADDRESS")
        .unwrap_or_else(|_| "[::1]:50051".to_string())
        .parse()?;

    info!("=====================================");
    info!("vLLM Rust Core KV Cache Server");
    info!("=====================================");
    info!("gRPC server address: {}", addr);
    info!("Number of blocks: {}", num_blocks);
    info!("Press Ctrl+C for graceful shutdown");
    info!("=====================================");

    // Create the service
    let enable_prefix_caching =
        std::env::var("ENABLE_PREFIX_CACHING").unwrap_or_else(|_| "false".to_string()) == "true";
    let service = KvCacheServiceImpl::new(num_blocks, enable_prefix_caching);
    let running = service.running.clone();

    // Setup graceful shutdown with multiple signals
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    // Spawn shutdown handler
    tokio::spawn(async move {
        // Wait for either SIGINT (Ctrl+C) or SIGTERM
        let ctrl_c = tokio::signal::ctrl_c();

        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm =
                signal(SignalKind::terminate()).expect("Failed to install SIGTERM handler");

            tokio::select! {
                _ = ctrl_c => {
                    info!("Received CTRL+C, initiating graceful shutdown");
                }
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                }
            }
        }

        #[cfg(not(unix))]
        {
            ctrl_c.await.expect("Failed to listen for CTRL+C");
            info!("Received CTRL+C, initiating graceful shutdown");
        }

        let _ = tx.send(());
    });

    // Start the gRPC server with graceful shutdown
    let server = Server::builder()
        .add_service(KvCacheServiceServer::new(service))
        .serve_with_shutdown(addr, async {
            rx.await.ok();
            info!("Shutdown signal received, stopping server...");
        });

    // Run the server
    let result = match server.await {
        Ok(_) => {
            info!("Server shutdown complete");
            Ok(())
        }
        Err(e) => {
            error!("Server error: {}", e);
            Err(Box::new(e) as Box<dyn std::error::Error>)
        }
    };

    // Ensure clean exit
    info!("Performing final cleanup...");
    info!("Goodbye!");

    result
}

/// Cleanup handler for unexpected termination
impl Drop for KvCacheServiceImpl {
    fn drop(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            warn!("Service dropping while still running - performing emergency cleanup");
            self.running.store(false, Ordering::SeqCst);

            // Try to send shutdown command to KV manager thread
            // Note: We can't use blocking_send here as we might be in an async context
            let tx = self.command_tx.clone();
            std::thread::spawn(move || {
                let _ = tx.blocking_send(KvCommand::Shutdown);
            });
            info!("Initiated shutdown of KV manager thread");
        }
    }
}
