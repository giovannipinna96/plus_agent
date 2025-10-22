#!/bin/bash
# Helper script to run ONLY multi-agent benchmark
# Usage: ./bash/run_benchmark_multi.sh

echo "Launching Multi-Agent benchmark ONLY..."
sbatch --export=BENCH_MODE=multi bash/test_smolagents.slurm
