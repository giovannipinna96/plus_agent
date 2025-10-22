#!/bin/bash
# Helper script to run ONLY single-agent benchmark
# Usage: ./bash/run_benchmark_single.sh

echo "Launching Single-Agent benchmark ONLY..."
sbatch --export=BENCH_MODE=single bash/test_smolagents.slurm
