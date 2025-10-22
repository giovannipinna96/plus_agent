#!/usr/bin/env python3
"""
Benchmark Script for SmolaAgents SingleAgent2 Pipeline
======================================================

This script runs the smolagents_singleagent2 pipeline 10 times
on each of the 10 Titanic questions, with complete resource cleanup between runs.

Usage:
    uv run python benchmark_singleagent2.py
"""

import gc
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import psutil
import os

# Import the singleagent2 pipeline module
import smolagents_singleagent2 as singleagent2

# Import questions from JSON (detailed version)
import json

# Load detailed Titanic questions
with open('titanic_questions_detailed.json', 'r', encoding='utf-8') as f:
    TITANIC_QUESTIONS_DETAILED_DICT = json.load(f)

# Convert to list format for compatibility
TITANIC_QUESTIONS = []
for i in range(1, 11):
    q_key = f"question_{i}"
    if q_key in TITANIC_QUESTIONS_DETAILED_DICT:
        q_data = TITANIC_QUESTIONS_DETAILED_DICT[q_key]
        TITANIC_QUESTIONS.append({
            "numero": i,
            "livello": q_data.get("complexity", "unknown").upper(),
            "domanda": q_data.get("text", ""),  # Use detailed text instead of simple question
            "complexity": q_data.get("complexity", "unknown"),
            "description": q_data.get("description", "")
        })


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
        pipeline_func: Function to run (singleagent2.run_analysis)
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
        pipeline_name: Name of the pipeline ("singleagent2")
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

def main():
    """Main benchmark execution for singleagent2 only."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║            🔬 SMOLAGENTS SINGLEAGENT2 BENCHMARK SUITE                      ║
║                                                                            ║
║          Comprehensive evaluation of singleagent2 pipeline                ║
║                    on Titanic dataset                                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    total_experiments = CONFIG['num_runs'] * len(TITANIC_QUESTIONS)

    print("📋 Benchmark Configuration:")
    print(f"   • Pipeline: smolagents_singleagent2 ONLY")
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

    # ========================================================================
    # BENCHMARK: SingleAgent2 Pipeline
    # ========================================================================

    print("\n" + "="*80)
    print("STARTING: SingleAgent2 Pipeline Benchmark")
    print("="*80)

    singleagent2_results = run_pipeline_benchmark(
        pipeline_name="singleagent2",
        pipeline_func=singleagent2.run_analysis,
        questions=TITANIC_QUESTIONS
    )

    # Save results
    save_results(
        singleagent2_results,
        f"singleagent2_benchmark_{timestamp}.json"
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    overall_time = time.time() - overall_start_time

    combined_results = {
        "benchmark_metadata": {
            "timestamp": timestamp,
            "pipeline": "singleagent2",
            "total_duration_seconds": overall_time,
            "config": CONFIG.copy()
        },
        "singleagent2": singleagent2_results
    }

    save_results(
        combined_results,
        f"singleagent2_full_benchmark_{timestamp}.json"
    )

    print("\n" + "="*80)
    print("🎉 BENCHMARK COMPLETED!")
    print("="*80)
    print(f"\n⏱️  Total Duration: {overall_time/60:.2f} minutes ({overall_time:.2f} seconds)")

    stats = singleagent2_results["overall_statistics"]
    print(f"\n📊 SINGLEAGENT2 SUMMARY:")
    print(f"{'─'*80}")
    print(f"   Success Rate: {stats['overall_success_rate']*100:.1f}%")
    print(f"   OOM Rate: {stats['overall_oom_rate']*100:.1f}%")
    if stats['avg_time_seconds']:
        print(f"   Average Time: {stats['avg_time_seconds']:.2f}s")
        print(f"   Total Time: {stats['total_time_seconds']:.2f}s")
    print(f"{'─'*80}\n")

    if stats['total_oom_errors'] > 0:
        print(f"⚠️  OUT OF MEMORY SUMMARY:")
        print(f"   SingleAgent2: {stats['total_oom_errors']} OOM errors")
        print()

    print(f"📁 All results saved to: {Path(CONFIG['results_dir']).absolute()}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
