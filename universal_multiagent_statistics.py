"""
Universal Multi-Agent Statistics Framework
A dataset-agnostic and request-independent system for collecting and analyzing multi-agent performance.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import statistics
from collections import defaultdict, Counter


@dataclass
class ToolExecution:
    """Represents a single tool execution."""
    tool_name: str
    agent_name: str
    parameters: Dict[str, Any]
    start_time: float
    end_time: float
    execution_time: float
    success: bool
    result_preview: str
    result_size: int
    error_message: Optional[str] = None


@dataclass
class AgentActivation:
    """Represents an agent activation."""
    agent_name: str
    start_time: float
    end_time: float
    execution_time: float
    tools_used: List[str]
    iterations: int
    success: bool
    result_summary: str
    error_message: Optional[str] = None


@dataclass
class SessionMetrics:
    """Complete metrics for a session."""
    session_id: str
    request_text: str
    start_time: float
    end_time: float
    total_time: float
    success: bool

    # Execution details
    tools_executed: List[ToolExecution]
    agents_activated: List[AgentActivation]

    # Summary statistics
    total_tools: int
    total_agents: int
    unique_tools: int
    unique_agents: int

    # Performance metrics
    avg_tool_time: float
    avg_agent_time: float
    success_rate: float

    # Optional metadata
    metadata: Dict[str, Any]


class UniversalStatsCollector:
    """Universal statistics collector for multi-agent systems."""

    def __init__(self):
        self.sessions: List[SessionMetrics] = []
        self.current_session: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()

    def start_session(self, request_text: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new tracking session."""
        session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        with self.lock:
            self.current_session = {
                "session_id": session_id,
                "request_text": request_text,
                "start_time": time.time(),
                "metadata": metadata or {},
                "tools_executed": [],
                "agents_activated": [],
                "success": False
            }

        return session_id

    def log_tool_execution(self, tool_name: str, agent_name: str, parameters: Dict[str, Any],
                          execution_time: float, success: bool, result: Any = None,
                          error: str = None) -> None:
        """Log a tool execution."""
        with self.lock:
            if self.current_session:
                result_str = str(result) if result else ""
                tool_exec = ToolExecution(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    parameters=parameters,
                    start_time=time.time() - execution_time,
                    end_time=time.time(),
                    execution_time=execution_time,
                    success=success,
                    result_preview=result_str[:200],
                    result_size=len(result_str),
                    error_message=error
                )
                self.current_session["tools_executed"].append(tool_exec)

    def log_agent_activation(self, agent_name: str, execution_time: float,
                           tools_used: List[str], iterations: int = 1,
                           success: bool = True, result_summary: str = "",
                           error: str = None) -> None:
        """Log an agent activation."""
        with self.lock:
            if self.current_session:
                agent_activation = AgentActivation(
                    agent_name=agent_name,
                    start_time=time.time() - execution_time,
                    end_time=time.time(),
                    execution_time=execution_time,
                    tools_used=tools_used,
                    iterations=iterations,
                    success=success,
                    result_summary=result_summary[:200],
                    error_message=error
                )
                self.current_session["agents_activated"].append(agent_activation)

    def end_session(self, success: bool = True) -> SessionMetrics:
        """End the current session and calculate metrics."""
        with self.lock:
            if not self.current_session:
                raise ValueError("No active session to end")

            session = self.current_session
            session["end_time"] = time.time()
            session["success"] = success

            # Calculate metrics
            metrics = self._calculate_session_metrics(session)
            self.sessions.append(metrics)
            self.current_session = None

            return metrics

    def _calculate_session_metrics(self, session: Dict[str, Any]) -> SessionMetrics:
        """Calculate comprehensive metrics for a session."""
        tools_executed = session["tools_executed"]
        agents_activated = session["agents_activated"]

        # Basic counts
        total_tools = len(tools_executed)
        total_agents = len(agents_activated)
        unique_tools = len(set(tool.tool_name for tool in tools_executed))
        unique_agents = len(set(agent.agent_name for agent in agents_activated))

        # Performance metrics
        tool_times = [tool.execution_time for tool in tools_executed if tool.success]
        agent_times = [agent.execution_time for agent in agents_activated if agent.success]

        avg_tool_time = statistics.mean(tool_times) if tool_times else 0.0
        avg_agent_time = statistics.mean(agent_times) if agent_times else 0.0

        # Success rate
        successful_ops = sum(1 for tool in tools_executed if tool.success) + \
                        sum(1 for agent in agents_activated if agent.success)
        total_ops = total_tools + total_agents
        success_rate = successful_ops / total_ops if total_ops > 0 else 1.0

        return SessionMetrics(
            session_id=session["session_id"],
            request_text=session["request_text"],
            start_time=session["start_time"],
            end_time=session["end_time"],
            total_time=session["end_time"] - session["start_time"],
            success=session["success"],
            tools_executed=tools_executed,
            agents_activated=agents_activated,
            total_tools=total_tools,
            total_agents=total_agents,
            unique_tools=unique_tools,
            unique_agents=unique_agents,
            avg_tool_time=avg_tool_time,
            avg_agent_time=avg_agent_time,
            success_rate=success_rate,
            metadata=session["metadata"]
        )


class UniversalStatsAnalyzer:
    """Universal analyzer for multi-agent statistics."""

    def __init__(self, sessions: List[SessionMetrics]):
        self.sessions = sessions

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance."""
        if not self.sessions:
            return {"error": "No sessions to analyze"}

        # Basic performance metrics
        total_sessions = len(self.sessions)
        successful_sessions = sum(1 for s in self.sessions if s.success)
        success_rate = successful_sessions / total_sessions * 100

        # Timing analysis
        session_times = [s.total_time for s in self.sessions]
        avg_session_time = statistics.mean(session_times)

        # Tool usage analysis
        all_tools = []
        for session in self.sessions:
            all_tools.extend([t.tool_name for t in session.tools_executed])

        tool_usage = Counter(all_tools)

        # Agent usage analysis
        all_agents = []
        for session in self.sessions:
            all_agents.extend([a.agent_name for a in session.agents_activated])

        agent_usage = Counter(all_agents)

        return {
            "session_metrics": {
                "total_sessions": total_sessions,
                "successful_sessions": successful_sessions,
                "success_rate": success_rate,
                "average_session_time": avg_session_time,
                "min_session_time": min(session_times),
                "max_session_time": max(session_times),
                "total_processing_time": sum(session_times)
            },
            "tool_usage": {
                "unique_tools": len(tool_usage),
                "total_tool_calls": sum(tool_usage.values()),
                "most_used_tools": tool_usage.most_common(10),
                "tools_per_session": sum(tool_usage.values()) / total_sessions
            },
            "agent_usage": {
                "unique_agents": len(agent_usage),
                "total_agent_activations": sum(agent_usage.values()),
                "most_used_agents": agent_usage.most_common(10),
                "agents_per_session": sum(agent_usage.values()) / total_sessions
            }
        }

    def analyze_efficiency(self) -> Dict[str, Any]:
        """Analyze system efficiency metrics."""
        if not self.sessions:
            return {"error": "No sessions to analyze"}

        # Tool efficiency
        tool_times = defaultdict(list)
        tool_success_rates = defaultdict(list)

        for session in self.sessions:
            for tool in session.tools_executed:
                tool_times[tool.tool_name].append(tool.execution_time)
                tool_success_rates[tool.tool_name].append(tool.success)

        tool_efficiency = {}
        for tool_name, times in tool_times.items():
            success_rate = sum(tool_success_rates[tool_name]) / len(tool_success_rates[tool_name])
            tool_efficiency[tool_name] = {
                "average_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "success_rate": success_rate,
                "total_calls": len(times),
                "efficiency_score": success_rate / statistics.mean(times) if statistics.mean(times) > 0 else 0
            }

        # Agent efficiency
        agent_times = defaultdict(list)
        agent_success_rates = defaultdict(list)

        for session in self.sessions:
            for agent in session.agents_activated:
                agent_times[agent.agent_name].append(agent.execution_time)
                agent_success_rates[agent.agent_name].append(agent.success)

        agent_efficiency = {}
        for agent_name, times in agent_times.items():
            success_rate = sum(agent_success_rates[agent_name]) / len(agent_success_rates[agent_name])
            agent_efficiency[agent_name] = {
                "average_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "success_rate": success_rate,
                "total_activations": len(times),
                "efficiency_score": success_rate / statistics.mean(times) if statistics.mean(times) > 0 else 0
            }

        return {
            "tool_efficiency": tool_efficiency,
            "agent_efficiency": agent_efficiency,
            "overall_efficiency": {
                "best_performing_tool": max(tool_efficiency.items(), key=lambda x: x[1]["efficiency_score"])[0] if tool_efficiency else None,
                "best_performing_agent": max(agent_efficiency.items(), key=lambda x: x[1]["efficiency_score"])[0] if agent_efficiency else None
            }
        }

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns and workflows."""
        if not self.sessions:
            return {"error": "No sessions to analyze"}

        # Workflow patterns
        workflow_patterns = []
        for session in self.sessions:
            agents_sequence = [a.agent_name for a in session.agents_activated]
            tools_sequence = [t.tool_name for t in session.tools_executed]

            workflow_patterns.append({
                "session_id": session.session_id,
                "agent_sequence": agents_sequence,
                "tool_sequence": tools_sequence,
                "total_time": session.total_time,
                "success": session.success
            })

        # Common patterns
        agent_sequences = [tuple(p["agent_sequence"]) for p in workflow_patterns]
        common_agent_patterns = Counter(agent_sequences).most_common(5)

        # Time patterns
        time_categories = {
            "fast": [p for p in workflow_patterns if p["total_time"] < 1.0],
            "medium": [p for p in workflow_patterns if 1.0 <= p["total_time"] < 5.0],
            "slow": [p for p in workflow_patterns if p["total_time"] >= 5.0]
        }

        return {
            "workflow_patterns": workflow_patterns,
            "common_agent_patterns": [{"pattern": list(pattern), "count": count}
                                    for pattern, count in common_agent_patterns],
            "time_distribution": {
                category: len(patterns) for category, patterns in time_categories.items()
            },
            "pattern_insights": {
                "total_unique_workflows": len(set(agent_sequences)),
                "most_common_workflow_usage": common_agent_patterns[0][1] / len(workflow_patterns) * 100 if common_agent_patterns else 0
            }
        }

    def generate_recommendations(self) -> List[str]:
        """Generate system improvement recommendations."""
        performance = self.analyze_performance()
        efficiency = self.analyze_efficiency()
        patterns = self.analyze_patterns()

        recommendations = []

        # Performance-based recommendations
        success_rate = performance["session_metrics"]["success_rate"]
        if success_rate < 95:
            recommendations.append(f"Improve system reliability - current success rate is {success_rate:.1f}%")

        avg_time = performance["session_metrics"]["average_session_time"]
        if avg_time > 10:
            recommendations.append(f"Optimize performance - average session time is {avg_time:.2f} seconds")

        # Efficiency-based recommendations
        tool_efficiency = efficiency.get("tool_efficiency", {})
        inefficient_tools = [name for name, stats in tool_efficiency.items()
                           if stats["efficiency_score"] < 0.1]

        if inefficient_tools:
            recommendations.append(f"Review inefficient tools: {', '.join(inefficient_tools[:3])}")

        # Pattern-based recommendations
        unique_workflows = patterns["pattern_insights"]["total_unique_workflows"]
        total_sessions = len(self.sessions)

        if unique_workflows / total_sessions > 0.8:
            recommendations.append("High workflow diversity detected - consider workflow standardization")

        # Usage-based recommendations
        tools_per_session = performance["tool_usage"]["tools_per_session"]
        if tools_per_session < 1:
            recommendations.append("Low tool usage detected - verify tool integration")

        return recommendations


class UniversalStatsExporter:
    """Universal exporter for multi-agent statistics."""

    def __init__(self, collector: UniversalStatsCollector):
        self.collector = collector
        self.analyzer = UniversalStatsAnalyzer(collector.sessions)

    def export_comprehensive_stats(self, filename: str = None) -> str:
        """Export comprehensive statistics to JSON."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"universal_multiagent_stats_{timestamp}.json"

        # Generate all analyses
        performance = self.analyzer.analyze_performance()
        efficiency = self.analyzer.analyze_efficiency()
        patterns = self.analyzer.analyze_patterns()
        recommendations = self.analyzer.generate_recommendations()

        # Prepare export data
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "framework_version": "1.0.0",
                "total_sessions": len(self.collector.sessions),
                "framework_type": "Universal Multi-Agent Statistics"
            },
            "performance_analysis": performance,
            "efficiency_analysis": efficiency,
            "pattern_analysis": patterns,
            "recommendations": recommendations,
            "raw_sessions": [asdict(session) for session in self.collector.sessions]
        }

        # Export to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            return filename
        except Exception as e:
            raise Exception(f"Failed to export statistics: {str(e)}")

    def export_summary_report(self, filename: str = None) -> str:
        """Export a summary report in markdown format."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"universal_multiagent_report_{timestamp}.md"

        performance = self.analyzer.analyze_performance()
        efficiency = self.analyzer.analyze_efficiency()
        recommendations = self.analyzer.generate_recommendations()

        # Generate markdown report
        report = f"""# Universal Multi-Agent System Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sessions Analyzed:** {len(self.collector.sessions)}

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Sessions | {performance['session_metrics']['total_sessions']} |
| Success Rate | {performance['session_metrics']['success_rate']:.1f}% |
| Average Session Time | {performance['session_metrics']['average_session_time']:.2f}s |
| Total Processing Time | {performance['session_metrics']['total_processing_time']:.2f}s |

## Tool Usage

| Metric | Value |
|--------|-------|
| Unique Tools | {performance['tool_usage']['unique_tools']} |
| Total Tool Calls | {performance['tool_usage']['total_tool_calls']} |
| Tools per Session | {performance['tool_usage']['tools_per_session']:.1f} |

## Agent Usage

| Metric | Value |
|--------|-------|
| Unique Agents | {performance['agent_usage']['unique_agents']} |
| Total Agent Activations | {performance['agent_usage']['total_agent_activations']} |
| Agents per Session | {performance['agent_usage']['agents_per_session']:.1f} |

## Recommendations

"""

        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        # Export to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            return filename
        except Exception as e:
            raise Exception(f"Failed to export report: {str(e)}")


# Global instance for easy use
global_stats_collector = UniversalStatsCollector()


def track_multiagent_session(request_text: str, metadata: Dict[str, Any] = None):
    """Context manager for tracking a multi-agent session."""
    class SessionTracker:
        def __init__(self, request_text: str, metadata: Dict[str, Any] = None):
            self.request_text = request_text
            self.metadata = metadata
            self.session_id = None

        def __enter__(self):
            self.session_id = global_stats_collector.start_session(self.request_text, self.metadata)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            success = exc_type is None
            global_stats_collector.end_session(success)

        def log_tool(self, tool_name: str, agent_name: str, parameters: Dict[str, Any],
                    execution_time: float, success: bool, result: Any = None, error: str = None):
            global_stats_collector.log_tool_execution(tool_name, agent_name, parameters,
                                                    execution_time, success, result, error)

        def log_agent(self, agent_name: str, execution_time: float, tools_used: List[str],
                     iterations: int = 1, success: bool = True, result_summary: str = "", error: str = None):
            global_stats_collector.log_agent_activation(agent_name, execution_time, tools_used,
                                                       iterations, success, result_summary, error)

    return SessionTracker(request_text, metadata)


def export_universal_stats(filename: str = None) -> str:
    """Export universal statistics."""
    exporter = UniversalStatsExporter(global_stats_collector)
    return exporter.export_comprehensive_stats(filename)


def export_universal_report(filename: str = None) -> str:
    """Export universal report."""
    exporter = UniversalStatsExporter(global_stats_collector)
    return exporter.export_summary_report(filename)


# Example usage
if __name__ == "__main__":
    # Example of how to use the universal statistics system

    # Track a session
    with track_multiagent_session("Analyze customer data", {"dataset": "customers", "complexity": "medium"}) as tracker:
        # Log tool usage
        tracker.log_tool("data_reader", "DataAgent", {"file": "data.csv"}, 0.5, True, "Data loaded successfully")
        tracker.log_tool("aggregator", "AnalysisAgent", {"column": "age"}, 0.3, True, "Aggregation complete")

        # Log agent activation
        tracker.log_agent("DataAgent", 1.2, ["data_reader"], 1, True, "Data processing completed")
        tracker.log_agent("AnalysisAgent", 0.8, ["aggregator"], 1, True, "Analysis completed")

    # Export statistics
    stats_file = export_universal_stats()
    report_file = export_universal_report()

    print(f"Statistics exported to: {stats_file}")
    print(f"Report exported to: {report_file}")