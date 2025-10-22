#!/bin/bash
#SBATCH --job-name=smolagent2
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:a100
#SBATCH --time=43:00:00
#SBATCH --output=logs/test_smolagents_singleagent2_%j.out
#SBATCH --error=logs/test_smolagents_singleagent2_%j.err

# ============================================================================
# SmolaAgents SingleAgent2 Benchmark Script
# ============================================================================
# This script runs benchmark ONLY for smolagents_singleagent2.py
#
# Usage:
#   sbatch bash/test_smolagents_singleagent2.slurm
#
# Description:
#   Runs 10 repetitions of all 10 Titanic questions using smolagents_singleagent2
#   Total: 100 experiments (10 questions × 10 runs)
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SmolaAgents SingleAgent2 Benchmark Execution           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Benchmark Configuration:"
echo "  • Pipeline: smolagents_singleagent2 ONLY"
echo "  • Model: openai/gpt-oss-120b"
echo "  • Questions: 10 (from titanic_questions.py)"
echo "  • Repetitions per question: 10"
echo "  • Total experiments: 100"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Set environment variables
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

cd ..

# Run benchmark for singleagent2 only
echo "🚀 Starting smolagents_singleagent2 benchmark"
echo ""

uv run python benchmark_singleagent2.py

BENCHMARK_EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Benchmark completed with exit code: $BENCHMARK_EXIT_CODE"
echo "End time: $(date)"
echo "════════════════════════════════════════════════════════════════"

exit $BENCHMARK_EXIT_CODE
