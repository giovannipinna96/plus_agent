#!/bin/bash
# Helper script to run BOTH single-agent and multi-agent benchmarks
# Usage: ./bash/run_benchmark_both.sh

echo "Launching BOTH Single-Agent and Multi-Agent benchmarks..."
sbatch --export=BENCH_MODE=both bash/test_smolagents.slurm
