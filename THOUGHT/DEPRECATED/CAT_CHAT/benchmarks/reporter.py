"""
Benchmark Reporter - Phase I.2
==============================

Generates human-readable comparison reports from benchmark results.

Output formats:
- Markdown reports
- JSON summaries
- Console tables
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from .runner import BenchmarkResult


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComparisonReport:
    """
    Structured comparison report for benchmark results.
    """
    scenario_name: str
    generated_at: str
    catalytic_result: Dict[str, Any]
    baseline_result: Dict[str, Any]
    comparison: Dict[str, Any]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        return BenchmarkReporter.format_markdown(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "generated_at": self.generated_at,
            "catalytic": self.catalytic_result,
            "baseline": self.baseline_result,
            "comparison": self.comparison,
        }


# =============================================================================
# Benchmark Reporter
# =============================================================================

class BenchmarkReporter:
    """
    Generates reports from benchmark results.

    Usage:
        reporter = BenchmarkReporter()

        # Generate comparison report
        report = reporter.create_comparison_report(catalytic, baseline)

        # Save as markdown
        reporter.save_markdown(report, output_path)

        # Print to console
        print(report.to_markdown())
    """

    @staticmethod
    def create_comparison_report(
        catalytic: BenchmarkResult,
        baseline: BenchmarkResult,
    ) -> ComparisonReport:
        """
        Create a comparison report from two benchmark results.

        Args:
            catalytic: Results from catalytic mode
            baseline: Results from baseline mode

        Returns:
            ComparisonReport with analysis
        """
        comparison = BenchmarkReporter._compute_comparison(catalytic, baseline)

        return ComparisonReport(
            scenario_name=catalytic.scenario_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            catalytic_result=catalytic.to_dict(),
            baseline_result=baseline.to_dict(),
            comparison=comparison,
        )

    @staticmethod
    def _compute_comparison(
        catalytic: BenchmarkResult,
        baseline: BenchmarkResult,
    ) -> Dict[str, Any]:
        """Compute detailed comparison metrics."""
        def safe_div(a, b):
            return a / b if b > 0 else 0

        def pct_change(new, old):
            if old == 0:
                return 0
            return ((new - old) / old) * 100

        return {
            "summary": {
                "compression_ratio": catalytic.compression_ratio,
                "token_reduction_pct": pct_change(
                    catalytic.peak_context_tokens,
                    baseline.peak_context_tokens
                ),
                "recall_delta": catalytic.recall_rate - baseline.recall_rate,
                "latency_improvement_pct": pct_change(
                    catalytic.total_latency_ms,
                    baseline.total_latency_ms
                ),
            },
            "tokens": {
                "catalytic_peak": catalytic.peak_context_tokens,
                "baseline_peak": baseline.peak_context_tokens,
                "saved": baseline.peak_context_tokens - catalytic.peak_context_tokens,
                "reduction_pct": abs(pct_change(
                    catalytic.peak_context_tokens,
                    baseline.peak_context_tokens
                )),
            },
            "bytes": {
                "catalytic_stored": catalytic.bytes_stored,
                "baseline_stored": baseline.bytes_stored,
                "saved": baseline.bytes_stored - catalytic.bytes_stored,
                "compression_ratio": catalytic.compression_ratio,
            },
            "quality": {
                "catalytic_recall": catalytic.recall_rate,
                "baseline_recall": baseline.recall_rate,
                "e_score_mean": catalytic.e_score_mean,
                "acceptable": catalytic.recall_rate >= 0.8,  # 80% threshold
            },
            "performance": {
                "catalytic_latency_ms": catalytic.total_latency_ms,
                "baseline_latency_ms": baseline.total_latency_ms,
                "catalytic_faster": catalytic.total_latency_ms < baseline.total_latency_ms,
            },
        }

    @staticmethod
    def format_markdown(report: ComparisonReport) -> str:
        """Format report as markdown."""
        comp = report.comparison
        cat = report.catalytic_result
        base = report.baseline_result

        lines = [
            "# Benchmark Comparison Report",
            "",
            f"**Scenario:** {report.scenario_name}",
            f"**Generated:** {report.generated_at}",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Metric | Baseline | Catalytic | Improvement |",
            "|--------|----------|-----------|-------------|",
            f"| Peak Context Tokens | {comp['tokens']['baseline_peak']:,} | {comp['tokens']['catalytic_peak']:,} | {comp['tokens']['reduction_pct']:.1f}% reduction |",
            f"| Total Bytes | {comp['bytes']['baseline_stored']:,} | {comp['bytes']['catalytic_stored']:,} | {comp['bytes']['compression_ratio']:.1f}x compression |",
            f"| Recall Rate | {comp['quality']['baseline_recall']:.1%} | {comp['quality']['catalytic_recall']:.1%} | {report.comparison['summary']['recall_delta']:+.1%} |",
            f"| Total Latency | {comp['performance']['baseline_latency_ms']:.0f}ms | {comp['performance']['catalytic_latency_ms']:.0f}ms | {'Faster' if comp['performance']['catalytic_faster'] else 'Slower'} |",
            "",
            "---",
            "",
            "## Compression Analysis",
            "",
            f"**Overall Compression Ratio:** {comp['bytes']['compression_ratio']:.1f}x",
            "",
            f"- Bytes expanded (input): {cat['compression']['bytes_expanded']:,}",
            f"- Bytes stored (catalytic): {cat['compression']['bytes_stored']:,}",
            f"- Bytes stored (baseline): {base['compression']['bytes_stored']:,}",
            f"- **Bytes saved:** {comp['bytes']['saved']:,} ({comp['tokens']['reduction_pct']:.1f}%)",
            "",
            "---",
            "",
            "## Quality Metrics",
            "",
            f"**Recall Rate:** {comp['quality']['catalytic_recall']:.1%} (target: 80%)",
            "",
            f"- E-score mean: {comp['quality']['e_score_mean']:.3f}",
            f"- Quality acceptable: {'Yes' if comp['quality']['acceptable'] else 'No'}",
            "",
        ]

        # Add recall details if available
        if cat.get('quality', {}).get('recall_details'):
            lines.extend([
                "### Recall Details",
                "",
                "| Fact ID | Turn | Recalled |",
                "|---------|------|----------|",
            ])
            for detail in cat['quality']['recall_details']:
                status = "Yes" if detail['recalled'] else "No"
                lines.append(f"| {detail['fact_id']} | {detail['turn']} | {status} |")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Performance Metrics",
            "",
            f"**Total Latency:**",
            f"- Catalytic: {comp['performance']['catalytic_latency_ms']:.0f}ms",
            f"- Baseline: {comp['performance']['baseline_latency_ms']:.0f}ms",
            "",
            f"**Turns Completed:**",
            f"- Catalytic: {cat.get('turns_completed', 'N/A')}",
            f"- Baseline: {base.get('turns_completed', 'N/A')}",
            "",
        ])

        # Add errors if any
        cat_errors = cat.get('errors', [])
        base_errors = base.get('errors', [])
        if cat_errors or base_errors:
            lines.extend([
                "---",
                "",
                "## Errors",
                "",
            ])
            if cat_errors:
                lines.append("**Catalytic errors:**")
                for err in cat_errors:
                    lines.append(f"- {err}")
                lines.append("")
            if base_errors:
                lines.append("**Baseline errors:**")
                for err in base_errors:
                    lines.append(f"- {err}")
                lines.append("")

        lines.extend([
            "---",
            "",
            "*Generated by CAT Chat Benchmark Reporter*",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_console(report: ComparisonReport) -> str:
        """Format report for console output."""
        comp = report.comparison

        width = 60
        sep = "=" * width

        lines = [
            sep,
            f"  BENCHMARK: {report.scenario_name}".center(width),
            sep,
            "",
            "  COMPRESSION",
            f"    Ratio:        {comp['bytes']['compression_ratio']:.1f}x",
            f"    Bytes saved:  {comp['bytes']['saved']:,}",
            f"    Token saved:  {comp['tokens']['saved']:,} ({comp['tokens']['reduction_pct']:.1f}%)",
            "",
            "  QUALITY",
            f"    Recall:       {comp['quality']['catalytic_recall']:.1%} (baseline: {comp['quality']['baseline_recall']:.1%})",
            f"    E-score:      {comp['quality']['e_score_mean']:.3f}",
            f"    Status:       {'PASS' if comp['quality']['acceptable'] else 'FAIL'}",
            "",
            "  PERFORMANCE",
            f"    Catalytic:    {comp['performance']['catalytic_latency_ms']:.0f}ms",
            f"    Baseline:     {comp['performance']['baseline_latency_ms']:.0f}ms",
            "",
            sep,
        ]

        return "\n".join(lines)

    def save_markdown(
        self,
        report: ComparisonReport,
        output_path: Path,
    ) -> None:
        """Save report as markdown file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

    def save_json(
        self,
        report: ComparisonReport,
        output_path: Path,
    ) -> None:
        """Save report as JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)


# =============================================================================
# Aggregate Reporting
# =============================================================================

class AggregateReporter:
    """
    Generates aggregate reports across multiple scenarios.
    """

    @staticmethod
    def create_summary(reports: List[ComparisonReport]) -> Dict[str, Any]:
        """
        Create aggregate summary across multiple benchmark runs.

        Args:
            reports: List of comparison reports

        Returns:
            Summary dictionary
        """
        if not reports:
            return {"error": "No reports provided"}

        compression_ratios = []
        recall_rates = []
        token_savings_pct = []

        for report in reports:
            comp = report.comparison
            compression_ratios.append(comp['bytes']['compression_ratio'])
            recall_rates.append(comp['quality']['catalytic_recall'])
            token_savings_pct.append(comp['tokens']['reduction_pct'])

        return {
            "scenarios_tested": len(reports),
            "compression": {
                "mean_ratio": sum(compression_ratios) / len(compression_ratios),
                "min_ratio": min(compression_ratios),
                "max_ratio": max(compression_ratios),
            },
            "quality": {
                "mean_recall": sum(recall_rates) / len(recall_rates),
                "min_recall": min(recall_rates),
                "all_acceptable": all(r >= 0.8 for r in recall_rates),
            },
            "tokens": {
                "mean_savings_pct": sum(token_savings_pct) / len(token_savings_pct),
            },
            "scenarios": [r.scenario_name for r in reports],
        }

    @staticmethod
    def format_summary_markdown(summary: Dict[str, Any]) -> str:
        """Format aggregate summary as markdown."""
        lines = [
            "# Aggregate Benchmark Summary",
            "",
            f"**Scenarios Tested:** {summary['scenarios_tested']}",
            "",
            "## Compression",
            "",
            f"- Mean ratio: {summary['compression']['mean_ratio']:.1f}x",
            f"- Range: {summary['compression']['min_ratio']:.1f}x - {summary['compression']['max_ratio']:.1f}x",
            "",
            "## Quality",
            "",
            f"- Mean recall: {summary['quality']['mean_recall']:.1%}",
            f"- Min recall: {summary['quality']['min_recall']:.1%}",
            f"- All acceptable (>80%): {'Yes' if summary['quality']['all_acceptable'] else 'No'}",
            "",
            "## Token Savings",
            "",
            f"- Mean savings: {summary['tokens']['mean_savings_pct']:.1f}%",
            "",
            "## Scenarios",
            "",
        ]

        for scenario in summary['scenarios']:
            lines.append(f"- {scenario}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_comparison_report(
    catalytic: BenchmarkResult,
    baseline: BenchmarkResult,
) -> ComparisonReport:
    """Generate a comparison report from benchmark results."""
    return BenchmarkReporter.create_comparison_report(catalytic, baseline)


__all__ = [
    "ComparisonReport",
    "BenchmarkReporter",
    "AggregateReporter",
    "generate_comparison_report",
]
