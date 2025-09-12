#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Throughput benchmark for KV cache operations: Python vs Rust implementations.

Note: This is a synthetic benchmark measuring theoretical maximum throughput
using immediate allocate/free cycles. Not representative of real vLLM usage patterns.
"""

import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

# ============================================================================
# Configuration
# ============================================================================

# Set up environment for VLLM V1
os.environ['VLLM_USE_V1'] = '1'
sys.path.insert(0, '/home/tahsintunan/Desktop/projects/vllm')


@dataclass
class BenchmarkConfig:
    """Configuration for the throughput benchmark."""
    
    # Core parameters
    batch_size: int = 1000              # Operations per batch
    total_operations: int = 100000      # Total operations to perform
    num_gpu_blocks: int = 50000         # Size of block pool
    enable_caching: bool = True         # Enable caching in block pool
    
    # Performance parameters
    warmup_operations: int = 10000      # Warmup operations before measurement
    measurement_runs: int = 5           # Number of measurement runs to average
    
    # Display parameters
    show_per_run_results: bool = False  # Show individual run results


# ============================================================================
# Benchmark Implementation
# ============================================================================

class ThroughputBenchmark:
    """Synthetic throughput benchmark measuring allocate/free cycle performance."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
    
    # ------------------------------------------------------------------------
    # Core Benchmark Methods
    # ------------------------------------------------------------------------
    
    def benchmark_python(self) -> Dict[str, float]:
        """Benchmark pure Python BlockPool implementation."""
        print("🐍 Python Benchmark")
        print(f"   Batch size: {self.config.batch_size}, Total ops: {self.config.total_operations:,}")
        
        from vllm.v1.core.block_pool import BlockPool

        # Setup (not measured)
        print("   Setup...")
        block_pool = BlockPool(
            num_gpu_blocks=self.config.num_gpu_blocks,
            enable_caching=self.config.enable_caching,
            enable_kv_cache_events=False
        )
        
        # Warmup (not measured)
        self._warmup(block_pool, "Python")
        
        # Actual measurements
        print(f"   Measuring ({self.config.measurement_runs} runs)...")
        run_times = []
        
        for run in range(self.config.measurement_runs):
            run_time = self._measure_operations(block_pool)
            run_times.append(run_time)
            
            if self.config.show_per_run_results:
                throughput = self.config.total_operations / run_time
                print(f"     Run {run + 1}: {throughput:,.0f} ops/s")
        
        return self._calculate_stats(run_times, is_rust=False)
    
    def benchmark_rust(self) -> Dict[str, float]:
        """Benchmark Rust KV cache server implementation."""
        print("🦀 Rust Benchmark")
        print(f"   Batch size: {self.config.batch_size}, Total ops: {self.config.total_operations:,}")
        
        from vllm.v1.core.kv_cache_backend_factory import (
            create_kv_cache_backend)

        # Setup (not measured) - includes server startup
        print("   Setup (starting server)...")
        os.environ['VLLM_KV_CACHE_BACKEND'] = 'rust'
        
        setup_start = time.perf_counter()
        backend = create_kv_cache_backend(
            num_gpu_blocks=self.config.num_gpu_blocks,
            enable_caching=self.config.enable_caching,
            enable_kv_cache_events=False
        )
        setup_time = time.perf_counter() - setup_start
        print(f"   Server startup: {setup_time:.3f}s (excluded from measurements)")
        
        # Warmup (not measured)
        self._warmup(backend, "Rust")
        
        # Actual measurements
        print(f"   Measuring ({self.config.measurement_runs} runs)...")
        run_times = []
        
        for run in range(self.config.measurement_runs):
            run_time = self._measure_operations(backend)
            run_times.append(run_time)
            
            if self.config.show_per_run_results:
                throughput = self.config.total_operations / run_time
                print(f"     Run {run + 1}: {throughput:,.0f} ops/s")
        
        # Cleanup (not measured)
        backend.rust_client.cleanup()
        
        stats = self._calculate_stats(run_times, is_rust=True)
        stats['setup_overhead_ms'] = setup_time * 1000
        
        return stats
    
    # ------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------
    
    def _warmup(self, backend, name: str) -> None:
        """Perform warmup to reach steady state."""
        warmup_batches = self.config.warmup_operations // self.config.batch_size
        print(f"   Warming up ({self.config.warmup_operations:,} operations)...")
        
        for _ in range(warmup_batches):
            blocks = backend.get_new_blocks(self.config.batch_size)
            backend.free_blocks(blocks)
    
    def _measure_operations(self, backend) -> float:
        """
        Measure time for allocate/free cycles.
        Note: Uses immediate free pattern for maximum throughput testing.
        """
        num_batches = self.config.total_operations // self.config.batch_size
        
        # Force garbage collection before measurement
        gc.collect()
        
        # High-precision timing
        start = time.perf_counter()
        
        for _ in range(num_batches):
            blocks = backend.get_new_blocks(self.config.batch_size)
            backend.free_blocks(blocks)
        
        end = time.perf_counter()
        
        return end - start
    
    def _calculate_stats(self, run_times: List[float], is_rust: bool = False) -> Dict[str, float]:
        """Calculate throughput and latency statistics."""
        # Throughput in ops/second
        throughputs = [self.config.total_operations / t for t in run_times]
        
        # Latency per operation in microseconds
        latencies_us = [(t / self.config.total_operations) * 1_000_000 for t in run_times]
        
        stats = {
            # Throughput metrics
            'mean_throughput': statistics.mean(throughputs),
            'std_throughput': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            
            # Latency metrics (per operation)
            'mean_latency_us': statistics.mean(latencies_us),
            'std_latency_us': statistics.stdev(latencies_us) if len(latencies_us) > 1 else 0,
            
            # Total time
            'mean_time_s': statistics.mean(run_times),
        }
        
        # IPC metrics for Rust
        if is_rust:
            num_batches = self.config.total_operations // self.config.batch_size
            stats['total_ipc_calls'] = num_batches * 2  # allocate + free
            stats['ops_per_ipc_roundtrip'] = self.config.batch_size / 2
        
        return stats
    
    # ------------------------------------------------------------------------
    # Main Benchmark Runner
    # ------------------------------------------------------------------------
    
    def run_comparison(self) -> None:
        """Run complete comparison benchmark."""
        print("=" * 80)
        print("THROUGHPUT BENCHMARK COMPARISON")
        print("=" * 80)
        print("Configuration:")
        print(f"  Batch size: {self.config.batch_size:,}")
        print(f"  Total operations: {self.config.total_operations:,}")
        print(f"  Block pool size: {self.config.num_gpu_blocks:,}")
        print(f"  Measurement runs: {self.config.measurement_runs}")
        print("=" * 80)
        
        # Run benchmarks
        self.results['python'] = self.benchmark_python()
        print()
        
        self.results['rust'] = self.benchmark_rust()
        print()
        
        # Display results
        self._display_results()
    
    def _display_results(self) -> None:
        """Display formatted benchmark results."""
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        # Main comparison table
        print(f"{'Implementation':<15} {'Throughput (ops/s)':<22} {'Latency (μs/op)':<18}")
        print("-" * 60)
        
        python_stats = self.results['python']
        rust_stats = self.results['rust']
        
        # Python results
        python_throughput = f"{python_stats['mean_throughput']:,.0f} ± {python_stats['std_throughput']:,.0f}"
        python_latency = f"{python_stats['mean_latency_us']:.2f} ± {python_stats['std_latency_us']:.2f}"
        print(f"{'Python':<15} {python_throughput:<22} {python_latency:<18}")
        
        # Rust results
        rust_throughput = f"{rust_stats['mean_throughput']:,.0f} ± {rust_stats['std_throughput']:,.0f}"
        rust_latency = f"{rust_stats['mean_latency_us']:.2f} ± {rust_stats['std_latency_us']:.2f}"
        print(f"{'Rust':<15} {rust_throughput:<22} {rust_latency:<18}")
        
        print()
        print("=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Calculate speedup and overhead
        speedup = rust_stats['mean_throughput'] / python_stats['mean_throughput']
        latency_diff = rust_stats['mean_latency_us'] - python_stats['mean_latency_us']
        
        print(f"Throughput Speedup:   {speedup:.2f}x")
        print(f"Latency Difference:   {latency_diff:+.2f} μs/op (IPC overhead)")
        
        if 'setup_overhead_ms' in rust_stats:
            print(f"Server Startup Time:  {rust_stats['setup_overhead_ms']:.1f} ms (one-time cost)")
        
        if 'total_ipc_calls' in rust_stats:
            print(f"IPC Round-trips:      {rust_stats['total_ipc_calls']:,}")
            print(f"Operations per IPC:   {rust_stats['ops_per_ipc_roundtrip']:.0f}")
            
            # Calculate IPC overhead per call
            total_ipc_overhead_us = latency_diff * self.config.total_operations
            overhead_per_ipc_us = total_ipc_overhead_us / rust_stats['total_ipc_calls']
            print(f"Overhead per IPC:     {overhead_per_ipc_us:.2f} μs")
        
        print()
        print("=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)
        
        if speedup > 1.0:
            print(f"✅ Rust is {speedup:.2f}x faster than Python")
        else:
            print(f"⚠️  Rust is {1/speedup:.2f}x slower than Python")
        
        if latency_diff > 0:
            print(f"📊 IPC adds {latency_diff:.2f} μs overhead per operation")
            if self.config.batch_size < 1000:
                print("💡 Consider increasing batch_size to amortize IPC overhead")
        else:
            print(f"🚀 Rust reduces latency by {abs(latency_diff):.2f} μs per operation")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the benchmark with configurable parameters."""
    
    config = BenchmarkConfig(
        # TODO: 697 is fast, 698 is terribly slow. Need to investigate.
        batch_size=697,
        total_operations=10000000,
        num_gpu_blocks=20480,
        enable_caching=True,
        warmup_operations=50000,
        measurement_runs=5,
        show_per_run_results=False
    )
    
    benchmark = ThroughputBenchmark(config)
    benchmark.run_comparison()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()