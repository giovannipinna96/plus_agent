#!/usr/bin/env python3
"""
Benchmark Script for SmolaAgents Pipelines
==========================================

This script runs single-agent and/or multi-agent pipelines 10 times
on each of the 10 Titanic questions, with complete resource cleanup between runs.

Usage:
    uv run python benchmark_smolagents.py [--mode {single,multi,both}]

Options:
    --mode single    Run only single-agent benchmark
    --mode multi     Run only multi-agent benchmark
    --mode both      Run both benchmarks (default)
"""

import gc
import json
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
import psutil
import os

# Import the two pipeline modules
import smolagents_singleagent as single_agent
import smolagents_multiagent_system as multi_agent

# Import questions
from titanic_questions import TITANIC_QUESTIONS


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "num_runs": 10,  # Number of runs per question
    "dataset_path": "data/titanic.csv",  # Default dataset path
    "results_dir": "benchmark_results",  # Directory to save results
    "cleanup_between_runs": True,  # Whether to cleanup between runs
    "verbose": True,  # Print detailed progress
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def cleanup_resources():
    """
    Aggressively cleanup resources between runs.
    This includes:
    - Python garbage collection
    - Clearing CUDA cache (if available)
    - Forcing memory cleanup
    """
    # Run garbage collection multiple times
    for _ in range(3):
        gc.collect()

    # Clear CUDA cache if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # Small delay to ensure cleanup
    time.sleep(1)


def save_results(results, filename):
    """Save results to JSON file."""
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {filepath}")


def is_out_of_memory_error(error_obj):
    """
    Check if an error is related to out of memory.

    Args:
        error_obj: Exception object

    Returns:
        Boolean indicating if it's an OOM error
    """
    error_str = str(error_obj).lower()
    oom_indicators = [
        "out of memory",
        "oom",
        "cuda out of memory",
        "cudaerror",
        "memoryerror",
        "cannot allocate memory",
        "allocation failed"
    ]
    return any(indicator in error_str for indicator in oom_indicators)


def aggressive_cleanup_after_oom():
    """
    Extra aggressive cleanup after OOM error.
    Performs multiple rounds of cleanup to free as much memory as possible.
    """
    print_progress("⚠️  OUT OF MEMORY detected - performing aggressive cleanup...", level=3)

    # Multiple rounds of GC
    for i in range(5):
        gc.collect()

    # Clear CUDA cache aggressively
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            # Try to reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print_progress(f"   Warning during CUDA cleanup: {e}", level=3)

    # Extra delay for system to stabilize
    time.sleep(3)

    print_progress("   ✓ Aggressive cleanup completed", level=3)


def print_progress(message, level=0):
    """Print formatted progress message."""
    if CONFIG["verbose"]:
        indent = "  " * level
        print(f"{indent}{message}")


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def run_single_experiment(pipeline_func, question, run_number):
    """
    Run a single experiment and collect metrics.

    Args:
        pipeline_func: Function to run (single_agent or multi_agent)
        question: Question dictionary from titanic_questions
        run_number: Current run number (1-10)

    Returns:
        Dictionary with results and metrics
    """
    result = {
        "run_number": run_number,
        "question_number": question["numero"],
        "question_text": question["domanda"],
        "success": False,
        "error": None,
        "error_type": None,
        "is_oom": False,
        "execution_time_seconds": None,
        "memory_before_mb": None,
        "memory_after_mb": None,
        "memory_delta_mb": None,
        "result_text": None,
        "timestamp": datetime.now().isoformat()
    }

    # Measure memory before
    result["memory_before_mb"] = get_memory_usage()

    # Run the experiment
    start_time = time.time()

    try:
        output = pipeline_func(
            user_prompt=question["domanda"],
            file_path=CONFIG["dataset_path"]
        )

        result["success"] = True
        result["result_text"] = str(output)

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        result["traceback"] = traceback.format_exc()

        # Check if it's an OOM error
        result["is_oom"] = is_out_of_memory_error(e)

        if result["is_oom"]:
            # Print OOM error prominently
            print_progress("", level=3)
            print_progress("╔" + "="*76 + "╗", level=3)
            print_progress("║" + " "*20 + "⚠️  OUT OF MEMORY ERROR ⚠️" + " "*21 + "║", level=3)
            print_progress("╚" + "="*76 + "╝", level=3)
            print_progress(f"Error: {str(e)[:200]}...", level=3)

            # Perform aggressive cleanup immediately
            aggressive_cleanup_after_oom()

    # Measure time and memory after
    result["execution_time_seconds"] = time.time() - start_time
    result["memory_after_mb"] = get_memory_usage()
    result["memory_delta_mb"] = result["memory_after_mb"] - result["memory_before_mb"]

    return result


def run_pipeline_benchmark(pipeline_name, pipeline_func, questions):
    """
    Run complete benchmark for a pipeline on all questions.

    Args:
        pipeline_name: Name of the pipeline ("single_agent" or "multi_agent")
        pipeline_func: Function to run
        questions: List of question dictionaries

    Returns:
        Dictionary with all benchmark results
    """
    print_progress(f"\n{'='*80}")
    print_progress(f"🚀 Starting Benchmark: {pipeline_name.upper()}")
    print_progress(f"{'='*80}\n")

    benchmark_results = {
        "pipeline_name": pipeline_name,
        "start_time": datetime.now().isoformat(),
        "config": CONFIG.copy(),
        "questions": []
    }

    # For each question
    for q_idx, question in enumerate(questions, 1):
        print_progress(f"\n{'─'*80}")
        print_progress(f"📋 Question {question['numero']}/{len(questions)}: {question['livello']}", level=1)
        print_progress(f"❓ {question['domanda'][:80]}...", level=1)
        print_progress(f"{'─'*80}")

        question_results = {
            "question_number": question["numero"],
            "question_text": question["domanda"],
            "question_level": question["livello"],
            "runs": []
        }

        # Run 10 times for this question
        for run_num in range(1, CONFIG["num_runs"] + 1):
            print_progress(f"▶️  Run {run_num}/{CONFIG['num_runs']}", level=2)

            # Run experiment
            run_result = run_single_experiment(pipeline_func, question, run_num)
            question_results["runs"].append(run_result)

            # Print result
            if run_result["success"]:
                print_progress(
                    f"✅ Success | Time: {run_result['execution_time_seconds']:.2f}s | "
                    f"Memory Δ: {run_result['memory_delta_mb']:+.1f} MB",
                    level=3
                )
            else:
                if run_result["is_oom"]:
                    print_progress(f"❌ FAILED - OUT OF MEMORY", level=3)
                    print_progress(f"   Continuing with next run...", level=3)
                else:
                    error_msg = run_result['error'][:100] if run_result['error'] else "Unknown error"
                    print_progress(f"❌ Failed | Error: {error_msg}...", level=3)

            # Cleanup resources between runs
            # Use aggressive cleanup if OOM occurred
            if run_num < CONFIG["num_runs"]:
                if run_result["is_oom"]:
                    # Already did aggressive cleanup in run_single_experiment
                    print_progress("   Extra delay after OOM...", level=3)
                    time.sleep(2)
                elif CONFIG["cleanup_between_runs"]:
                    print_progress("🧹 Cleaning up resources...", level=3)
                    cleanup_resources()

        # Calculate statistics for this question
        successful_runs = [r for r in question_results["runs"] if r["success"]]
        oom_runs = [r for r in question_results["runs"] if r.get("is_oom", False)]
        failed_runs = [r for r in question_results["runs"] if not r["success"]]

        if successful_runs:
            times = [r["execution_time_seconds"] for r in successful_runs]
            question_results["statistics"] = {
                "success_rate": len(successful_runs) / CONFIG["num_runs"],
                "avg_time_seconds": sum(times) / len(times),
                "min_time_seconds": min(times),
                "max_time_seconds": max(times),
                "total_successful": len(successful_runs),
                "total_failed": CONFIG["num_runs"] - len(successful_runs),
                "total_oom_errors": len(oom_runs),
                "oom_rate": len(oom_runs) / CONFIG["num_runs"]
            }
        else:
            question_results["statistics"] = {
                "success_rate": 0.0,
                "total_successful": 0,
                "total_failed": CONFIG["num_runs"],
                "total_oom_errors": len(oom_runs),
                "oom_rate": len(oom_runs) / CONFIG["num_runs"]
            }

        print_progress(f"\n📊 Question {question['numero']} Statistics:", level=2)
        print_progress(f"   Success Rate: {question_results['statistics']['success_rate']*100:.1f}%", level=2)
        if oom_runs:
            print_progress(f"   ⚠️  OOM Errors: {len(oom_runs)} ({len(oom_runs)/CONFIG['num_runs']*100:.1f}%)", level=2)
        if successful_runs:
            print_progress(
                f"   Avg Time: {question_results['statistics']['avg_time_seconds']:.2f}s "
                f"(min: {question_results['statistics']['min_time_seconds']:.2f}s, "
                f"max: {question_results['statistics']['max_time_seconds']:.2f}s)",
                level=2
            )

        benchmark_results["questions"].append(question_results)

        # Major cleanup between questions
        print_progress("\n🧹 Major cleanup between questions...", level=1)
        cleanup_resources()

    benchmark_results["end_time"] = datetime.now().isoformat()

    # Calculate overall statistics
    all_successful = []
    all_times = []
    all_oom = []
    for q in benchmark_results["questions"]:
        for r in q["runs"]:
            if r["success"]:
                all_successful.append(r)
                all_times.append(r["execution_time_seconds"])
            if r.get("is_oom", False):
                all_oom.append(r)

    total_runs = len(questions) * CONFIG["num_runs"]
    benchmark_results["overall_statistics"] = {
        "total_runs": total_runs,
        "total_successful": len(all_successful),
        "total_failed": total_runs - len(all_successful),
        "total_oom_errors": len(all_oom),
        "overall_success_rate": len(all_successful) / total_runs if total_runs > 0 else 0,
        "overall_oom_rate": len(all_oom) / total_runs if total_runs > 0 else 0,
        "avg_time_seconds": sum(all_times) / len(all_times) if all_times else None,
        "min_time_seconds": min(all_times) if all_times else None,
        "max_time_seconds": max(all_times) if all_times else None,
        "total_time_seconds": sum(all_times) if all_times else 0
    }

    print_progress(f"\n{'='*80}")
    print_progress(f"📊 OVERALL STATISTICS - {pipeline_name.upper()}")
    print_progress(f"{'='*80}")
    print_progress(f"Total Runs: {total_runs}", level=1)
    print_progress(f"Successful: {len(all_successful)} ({benchmark_results['overall_statistics']['overall_success_rate']*100:.1f}%)", level=1)
    print_progress(f"Failed: {total_runs - len(all_successful)}", level=1)
    if all_oom:
        print_progress(f"⚠️  OOM Errors: {len(all_oom)} ({benchmark_results['overall_statistics']['overall_oom_rate']*100:.1f}%)", level=1)
    if all_times:
        print_progress(f"Average Time: {benchmark_results['overall_statistics']['avg_time_seconds']:.2f}s", level=1)
        print_progress(f"Total Time: {benchmark_results['overall_statistics']['total_time_seconds']:.2f}s", level=1)
    print_progress(f"{'='*80}\n")

    return benchmark_results


# ============================================================================
# MAIN BENCHMARK EXECUTION
# ============================================================================

def main(mode="both"):
    """
    Main benchmark execution.

    Args:
        mode: Which benchmark to run - "single", "multi", or "both" (default)
    """
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                   🔬 SMOLAGENTS BENCHMARK SUITE                            ║
║                                                                            ║
║          Comprehensive evaluation of single-agent vs multi-agent          ║
║                    pipelines on Titanic dataset                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Determine number of pipelines to run
    num_pipelines = 1 if mode in ["single", "multi"] else 2
    total_experiments = CONFIG['num_runs'] * len(TITANIC_QUESTIONS) * num_pipelines

    print("📋 Benchmark Configuration:")
    print(f"   • Mode: {mode.upper()}")
    print(f"   • Number of runs per question: {CONFIG['num_runs']}")
    print(f"   • Total questions: {len(TITANIC_QUESTIONS)}")
    print(f"   • Total experiments: {total_experiments}")
    print(f"   • Dataset: {CONFIG['dataset_path']}")
    print(f"   • Results directory: {CONFIG['results_dir']}")
    print(f"   • Cleanup between runs: {CONFIG['cleanup_between_runs']}")
    print()

    # Verify dataset exists
    if not Path(CONFIG["dataset_path"]).exists():
        print(f"❌ ERROR: Dataset not found at {CONFIG['dataset_path']}")
        print("Please ensure the Titanic dataset is available.")
        return

    print("✅ Dataset verified")
    print()

    overall_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize result variables
    single_agent_results = None
    multi_agent_results = None

    # ========================================================================
    # BENCHMARK 1: Single Agent Pipeline
    # ========================================================================

    if mode in ["single", "both"]:
        print("\n" + "="*80)
        print("PHASE 1: Single Agent Pipeline Benchmark")
        print("="*80)

        single_agent_results = run_pipeline_benchmark(
            pipeline_name="single_agent",
            pipeline_func=single_agent.run_analysis,
            questions=TITANIC_QUESTIONS
        )

        # Save intermediate results
        save_results(
            single_agent_results,
            f"single_agent_benchmark_{timestamp}.json"
        )

        # Major cleanup before next pipeline (if running both)
        if mode == "both":
            print("\n🧹 Major cleanup before next pipeline...")
            cleanup_resources()
            time.sleep(5)  # Extra delay between pipelines

    # ========================================================================
    # BENCHMARK 2: Multi-Agent Pipeline
    # ========================================================================

    if mode in ["multi", "both"]:
        print("\n" + "="*80)
        print(f"PHASE {'2' if mode == 'both' else '1'}: Multi-Agent Pipeline Benchmark")
        print("="*80)

        multi_agent_results = run_pipeline_benchmark(
            pipeline_name="multi_agent",
            pipeline_func=multi_agent.run_analysis,
            questions=TITANIC_QUESTIONS
        )

        # Save intermediate results
        save_results(
            multi_agent_results,
            f"multi_agent_benchmark_{timestamp}.json"
        )

    # ========================================================================
    # COMBINED RESULTS AND COMPARISON
    # ========================================================================

    overall_time = time.time() - overall_start_time

    # Build results based on mode
    if mode == "both":
        combined_results = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "mode": mode,
                "total_duration_seconds": overall_time,
                "config": CONFIG.copy()
            },
            "single_agent": single_agent_results,
            "multi_agent": multi_agent_results,
            "comparison": {
                "single_agent_success_rate": single_agent_results["overall_statistics"]["overall_success_rate"],
                "multi_agent_success_rate": multi_agent_results["overall_statistics"]["overall_success_rate"],
                "single_agent_oom_rate": single_agent_results["overall_statistics"]["overall_oom_rate"],
                "multi_agent_oom_rate": multi_agent_results["overall_statistics"]["overall_oom_rate"],
                "single_agent_avg_time": single_agent_results["overall_statistics"]["avg_time_seconds"],
                "multi_agent_avg_time": multi_agent_results["overall_statistics"]["avg_time_seconds"],
            }
        }

        # Calculate speedup/comparison if both have successful runs
        if (single_agent_results["overall_statistics"]["avg_time_seconds"] and
            multi_agent_results["overall_statistics"]["avg_time_seconds"]):

            sa_time = single_agent_results["overall_statistics"]["avg_time_seconds"]
            ma_time = multi_agent_results["overall_statistics"]["avg_time_seconds"]

            combined_results["comparison"]["speedup_ratio"] = ma_time / sa_time
            combined_results["comparison"]["faster_pipeline"] = "single_agent" if sa_time < ma_time else "multi_agent"

        # Save combined results
        save_results(
            combined_results,
            f"combined_benchmark_{timestamp}.json"
        )
    else:
        # Single mode - save only the results that were run
        result = single_agent_results if mode == "single" else multi_agent_results
        combined_results = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "mode": mode,
                "total_duration_seconds": overall_time,
                "config": CONFIG.copy()
            },
            mode.replace("multi", "multi_agent").replace("single", "single_agent"): result
        }
        save_results(
            combined_results,
            f"{mode}_agent_only_benchmark_{timestamp}.json"
        )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("🎉 BENCHMARK COMPLETED!")
    print("="*80)
    print(f"\n⏱️  Total Duration: {overall_time/60:.2f} minutes ({overall_time:.2f} seconds)")

    if mode == "both":
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"{'─'*80}")
        print(f"\n{'Pipeline':<20} {'Success Rate':<15} {'OOM Rate':<15} {'Avg Time':<15}")
        print(f"{'─'*80}")

        # Single Agent
        sa_stats = single_agent_results["overall_statistics"]
        print(f"{'Single Agent':<20} {sa_stats['overall_success_rate']*100:>6.1f}% {' '*7} "
              f"{sa_stats['overall_oom_rate']*100:>6.1f}% {' '*6} "
              f"{sa_stats['avg_time_seconds'] if sa_stats['avg_time_seconds'] else 0:>6.2f}s")

        # Multi Agent
        ma_stats = multi_agent_results["overall_statistics"]
        print(f"{'Multi Agent':<20} {ma_stats['overall_success_rate']*100:>6.1f}% {' '*7} "
              f"{ma_stats['overall_oom_rate']*100:>6.1f}% {' '*6} "
              f"{ma_stats['avg_time_seconds'] if ma_stats['avg_time_seconds'] else 0:>6.2f}s")

        print(f"{'─'*80}\n")

        # OOM Summary
        total_sa_oom = sa_stats['total_oom_errors']
        total_ma_oom = ma_stats['total_oom_errors']
        if total_sa_oom > 0 or total_ma_oom > 0:
            print(f"⚠️  OUT OF MEMORY SUMMARY:")
            if total_sa_oom > 0:
                print(f"   Single Agent: {total_sa_oom} OOM errors")
            if total_ma_oom > 0:
                print(f"   Multi Agent: {total_ma_oom} OOM errors")
            print()

        if "speedup_ratio" in combined_results["comparison"]:
            ratio = combined_results["comparison"]["speedup_ratio"]
            faster = combined_results["comparison"]["faster_pipeline"]
            print(f"⚡ {faster.replace('_', ' ').title()} is {abs(ratio):.2f}x faster")
            print()
    else:
        # Single mode summary
        result = single_agent_results if mode == "single" else multi_agent_results
        stats = result["overall_statistics"]

        print(f"\n📊 {mode.upper()} AGENT SUMMARY:")
        print(f"{'─'*80}")
        print(f"   Success Rate: {stats['overall_success_rate']*100:.1f}%")
        print(f"   OOM Rate: {stats['overall_oom_rate']*100:.1f}%")
        if stats['avg_time_seconds']:
            print(f"   Average Time: {stats['avg_time_seconds']:.2f}s")
            print(f"   Total Time: {stats['total_time_seconds']:.2f}s")
        print(f"{'─'*80}\n")

    print(f"📁 All results saved to: {Path(CONFIG['results_dir']).absolute()}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark script for SmolaAgents pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both pipelines (default)
  uv run python benchmark_smolagents.py
  uv run python benchmark_smolagents.py --mode both

  # Run only single-agent benchmark
  uv run python benchmark_smolagents.py --mode single

  # Run only multi-agent benchmark
  uv run python benchmark_smolagents.py --mode multi
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi", "both"],
        default="both",
        help="Which pipeline(s) to benchmark: single (single-agent only), multi (multi-agent only), or both (default)"
    )

    args = parser.parse_args()

    try:
        main(mode=args.mode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
