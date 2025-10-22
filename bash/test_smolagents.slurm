#!/bin/bash
#SBATCH --job-name=smolagent
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100
#SBATCH --time=43:00:00
#SBATCH --output=logs/test_smolagents_%j.out
#SBATCH --error=logs/test_smolagents_%j.err

# ============================================================================
# SmolaAgents Benchmark Script
# ============================================================================
# This script runs benchmarks for smolagents pipelines.
#
# Usage:
#   sbatch bash/test_smolagents.slurm                    # Run both pipelines (default)
#   sbatch --export=BENCH_MODE=single bash/test_smolagents.slurm   # Single-agent only
#   sbatch --export=BENCH_MODE=multi bash/test_smolagents.slurm    # Multi-agent only
#   sbatch --export=BENCH_MODE=both bash/test_smolagents.slurm     # Both pipelines
#
# Or set BENCH_MODE in your environment before sbatch:
#   export BENCH_MODE=single
#   sbatch bash/test_smolagents.slurm
#
# Options:
#   BENCH_MODE=single   Run only smolagents_singleagent benchmark
#   BENCH_MODE=multi    Run only smolagents_multiagent_system benchmark
#   BENCH_MODE=both     Run both benchmarks (default)
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              SmolaAgents Benchmark Execution                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# Set benchmark mode (default to "both" if not specified)
if [ -z "$BENCH_MODE" ]; then
    BENCH_MODE="both"
    echo "⚠️  BENCH_MODE not set, using default: both"
else
    echo "✅ BENCH_MODE set to: $BENCH_MODE"
fi

# Validate BENCH_MODE
if [[ ! "$BENCH_MODE" =~ ^(single|multi|both)$ ]]; then
    echo "❌ ERROR: Invalid BENCH_MODE='$BENCH_MODE'"
    echo "   Valid options: single, multi, both"
    echo "   Using default: both"
    BENCH_MODE="both"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Benchmark Configuration:"
echo "  • Mode: $BENCH_MODE"
echo "  • Model: Qwen/Qwen3-Coder-30B-A3B-Instruct"
echo "  • Pipelines to run:"

case $BENCH_MODE in
    single)
        echo "    - smolagents_singleagent ONLY"
        ;;
    multi)
        echo "    - smolagents_multiagent_system ONLY"
        ;;
    both)
        echo "    - smolagents_singleagent"
        echo "    - smolagents_multiagent_system"
        ;;
esac

echo "════════════════════════════════════════════════════════════════"
echo ""

# Set environment variables
# export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"
# export DEVICE="auto"

cd ..

# Run benchmark with specified mode
echo "🚀 Starting benchmark with mode: $BENCH_MODE"
echo ""

uv run python benchmark_smolagents.py --mode $BENCH_MODE

BENCHMARK_EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Benchmark completed with exit code: $BENCHMARK_EXIT_CODE"
echo "End time: $(date)"
echo "════════════════════════════════════════════════════════════════"

exit $BENCHMARK_EXIT_CODE
