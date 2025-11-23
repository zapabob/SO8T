#!/usr/bin/env python3
"""
Critical Infrastructure ABC Benchmark for SO8T Models
Tests modela, AEGIS, AEGISalpha0.6 on Defense, Aerospace, Semiconductor, Infrastructure scenarios
"""

import subprocess
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

def run_ollama_command(model: str, prompt: str, timeout: int = 240) -> Tuple[str, float]:
    """Run ollama command with extended timeout for complex reasoning"""
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        start_time = time.time()
        result = subprocess.run(
            ['ollama', 'run', model, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env=env
        )
        response_time = time.time() - start_time

        if result.returncode == 0:
            return result.stdout.strip(), response_time
        else:
            return f"[ERROR] {result.stderr}", 0.0
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] {timeout}s exceeded", 0.0
    except Exception as e:
        return f"[ERROR] {e}", 0.0

class CriticalInfrastructureBenchmark:
    """Critical Infrastructure Benchmark for Defense, Aerospace, Semiconductor, Infrastructure"""

    def __init__(self):
        self.models = {
            'modela': 'modela:latest',
            'aegis': 'aegis-adjusted:latest',
            'aegis_alpha_0_6': 'aegis-adjusted-0.6:latest'
        }

        # Critical infrastructure test scenarios
        self.benchmarks = {
            'defense': self._get_defense_tests(),
            'aerospace': self._get_aerospace_tests(),
            'semiconductor': self._get_semiconductor_tests(),
            'infrastructure': self._get_infrastructure_tests()
        }

        self.output_dir = Path("_docs/benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_defense_tests(self) -> List[Dict]:
        """Defense and military scenarios"""
        return [
            {
                'category': 'threat_detection',
                'scenario': 'Radar Signal Analysis',
                'question': 'Analyze this radar signature: Frequency 10.5 GHz, pulse width 1.2μs, PRI 800μs. Classify as friendly, hostile, or unknown. Provide reasoning.',
                'expected_elements': ['frequency_analysis', 'pulse_characteristics', 'threat_assessment'],
                'criticality': 'high'
            },
            {
                'category': 'situational_awareness',
                'scenario': 'Battlefield Intelligence',
                'question': 'Multiple contacts detected: 3 aircraft at 35,000ft heading 270°, 2 surface vessels 15nm offshore. Assess tactical situation and recommend response priority.',
                'expected_elements': ['contact_analysis', 'threat_prioritization', 'response_recommendation'],
                'criticality': 'critical'
            },
            {
                'category': 'cyber_defense',
                'scenario': 'Network Intrusion Detection',
                'question': 'Unusual traffic pattern: 5000 SYN packets/second from single IP, followed by data exfiltration attempts. Is this a DDoS attack or APT? Justify your analysis.',
                'expected_elements': ['traffic_analysis', 'attack_classification', 'defense_strategy'],
                'criticality': 'high'
            }
        ]

    def _get_aerospace_tests(self) -> List[Dict]:
        """Aerospace and aviation scenarios"""
        return [
            {
                'category': 'trajectory_calculation',
                'scenario': 'Orbital Mechanics',
                'question': 'Satellite in 500km circular orbit. Calculate orbital period, ground track speed, and time to next ground station pass over Hawaii (20°N, 155°W).',
                'expected_elements': ['orbital_period', 'ground_speed', 'pass_calculation'],
                'criticality': 'high'
            },
            {
                'category': 'system_diagnostics',
                'scenario': 'Aircraft System Failure',
                'question': 'Aircraft reports: Left engine oil pressure low, hydraulic system A degraded, cabin altitude 15,000ft. Prioritize systems and recommend emergency procedures.',
                'expected_elements': ['system_prioritization', 'emergency_procedures', 'safety_assessment'],
                'criticality': 'critical'
            },
            {
                'category': 'mission_planning',
                'scenario': 'Drone Swarm Coordination',
                'question': 'Coordinate 5 UAVs for reconnaissance mission: Area 10km², altitude 500ft, wind 15kts from NW. Optimize flight paths and communication protocols.',
                'expected_elements': ['path_optimization', 'communication_strategy', 'risk_assessment'],
                'criticality': 'high'
            }
        ]

    def _get_semiconductor_tests(self) -> List[Dict]:
        """Semiconductor manufacturing scenarios"""
        return [
            {
                'category': 'defect_analysis',
                'scenario': 'Wafer Inspection',
                'question': 'SEM image shows: 3μm particle at coordinate (1250, 890), micro-crack 0.5μm at (2100, 1450). Classify defects and assess yield impact on 300mm wafer.',
                'expected_elements': ['defect_classification', 'yield_calculation', 'process_impact'],
                'criticality': 'high'
            },
            {
                'category': 'process_optimization',
                'scenario': 'Etch Process Control',
                'question': 'Etch rate: 145nm/min, uniformity ±2.1%, selectivity SiO2:Si = 15:1. Optimize recipe for 28nm gate patterning with 5% uniformity improvement target.',
                'expected_elements': ['process_optimization', 'uniformity_improvement', 'selectivity_analysis'],
                'criticality': 'medium'
            },
            {
                'category': 'design_verification',
                'scenario': 'Timing Analysis',
                'question': 'Critical path: 2.8ns, setup time 150ps, hold time 80ps. Verify timing closure for 3GHz processor core with statistical analysis.',
                'expected_elements': ['timing_verification', 'statistical_analysis', 'closure_assessment'],
                'criticality': 'high'
            }
        ]

    def _get_infrastructure_tests(self) -> List[Dict]:
        """Critical infrastructure scenarios"""
        return [
            {
                'category': 'structural_analysis',
                'scenario': 'Bridge Inspection',
                'question': 'Bridge span 200m, concrete beams show 2mm cracks at 15m intervals. Wind load 45kN/m². Assess structural integrity and recommend inspection frequency.',
                'expected_elements': ['integrity_assessment', 'load_analysis', 'maintenance_schedule'],
                'criticality': 'critical'
            },
            {
                'category': 'power_grid',
                'scenario': 'Grid Stability Analysis',
                'question': 'Power grid: Frequency 59.8Hz, voltage 342kV, load 85%. Sudden 500MW generation loss. Calculate system response and stability margin.',
                'expected_elements': ['frequency_response', 'stability_analysis', 'contingency_planning'],
                'criticality': 'critical'
            },
            {
                'category': 'transportation',
                'scenario': 'Railway Signaling',
                'question': 'High-speed rail: Speed 250km/h, block length 2km, braking distance 1.2km. Design fail-safe signaling system for 2-minute headway.',
                'expected_elements': ['signaling_design', 'safety_analysis', 'capacity_optimization'],
                'criticality': 'high'
            }
        ]

    def evaluate_critical_response(self, response: str, task: Dict) -> float:
        """Evaluate response quality for critical infrastructure scenarios"""
        if not response or response.startswith('[ERROR]') or response.startswith('[TIMEOUT]'):
            return 0.0

        score = 0.0
        response_lower = response.lower()

        # Base quality check
        if len(response.strip()) > 20:
            score += 0.1

        # Criticality-based scoring
        criticality = task.get('criticality', 'medium')
        criticality_multiplier = {'critical': 1.5, 'high': 1.2, 'medium': 1.0, 'low': 0.8}[criticality]

        # Expected elements analysis
        expected_elements = task.get('expected_elements', [])
        elements_found = 0

        for element in expected_elements:
            element_variants = [element, element.replace('_', ' '), element.replace('_', '-')]
            if any(variant.lower() in response_lower for variant in element_variants):
                elements_found += 1

        if expected_elements:
            element_score = (elements_found / len(expected_elements)) * 0.4
            score += element_score

        # Technical accuracy indicators
        technical_terms = {
            'defense': ['threat', 'radar', 'tactical', 'intelligence', 'cyber'],
            'aerospace': ['orbital', 'trajectory', 'avionics', 'mission', 'safety'],
            'semiconductor': ['wafer', 'defect', 'process', 'yield', 'timing'],
            'infrastructure': ['structural', 'stability', 'integrity', 'grid', 'safety']
        }

        category = task.get('category', '')
        if category in technical_terms:
            tech_terms_found = sum(1 for term in technical_terms[category] if term in response_lower)
            score += min(tech_terms_found * 0.05, 0.2)

        # Reasoning quality
        reasoning_indicators = ['analyze', 'assess', 'calculate', 'recommend', 'justify', 'therefore', 'because']
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        score += min(reasoning_count * 0.05, 0.2)

        # Safety and reliability considerations
        safety_terms = ['safety', 'risk', 'reliability', 'redundancy', 'fail-safe', 'contingency']
        safety_count = sum(1 for term in safety_terms if term in response_lower)
        score += min(safety_count * 0.05, 0.1)

        final_score = min(score * criticality_multiplier, 1.0)

        return final_score

    def run_critical_benchmark(self) -> Dict[str, Any]:
        """Run critical infrastructure benchmark"""
        print("[BENCHMARK] Starting Critical Infrastructure ABC Benchmark")
        print("=" * 80)
        print("Testing Defense, Aerospace, Semiconductor, and Infrastructure scenarios")
        print("=" * 80)

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models': list(self.models.keys()),
                'benchmarks': list(self.benchmarks.keys()),
                'test_type': 'critical_infrastructure'
            },
            'results': [],
            'summary': {}
        }

        total_start_time = time.time()

        for model_key, model_name in tqdm(self.models.items(), desc="Models"):
            print(f"\n[MODEL] Testing {model_key} ({model_name})")

            for benchmark_name, tasks in tqdm(self.benchmarks.items(), desc=f"{model_key} Benchmarks", leave=False):
                print(f"  [DOMAIN] {benchmark_name.upper()} - Critical Infrastructure")

                for task_idx, task in enumerate(tasks):
                    # Enhanced prompt for critical scenarios
                    system_context = self._get_system_context(benchmark_name)
                    prompt = f"""SYSTEM CONTEXT: {system_context}

CRITICAL SCENARIO: {task['scenario']}
QUESTION: {task['question']}

REQUIREMENTS:
- Provide technical accuracy and reasoning
- Consider safety and reliability implications
- Include quantitative analysis where applicable
- Recommend specific actions or decisions

RESPONSE:"""

                    # Run inference with extended timeout
                    response, response_time = run_ollama_command(model_name, prompt, timeout=240)

                    # Evaluate response
                    score = self.evaluate_critical_response(response, task)

                    result = {
                        'model': model_key,
                        'domain': benchmark_name,
                        'category': task['category'],
                        'scenario': task['scenario'],
                        'criticality': task['criticality'],
                        'question': task['question'],
                        'response': response,
                        'score': score,
                        'response_time': response_time,
                        'timestamp': datetime.now().isoformat()
                    }

                    results['results'].append(result)

                    # Display progress
                    status = "[OK]" if score > 0.5 else "[NG]"
                    print(f"    {status} {task['category']}: {score:.2f} ({response_time:.1f}s)")

                    # Longer delay for critical infrastructure scenarios
                    time.sleep(1.0)

        total_time = time.time() - total_start_time
        results['metadata']['total_execution_time'] = total_time

        # Calculate comprehensive statistics
        results['summary'] = self.calculate_critical_statistics(results['results'])

        print(f"[BENCHMARK] Completed in {total_time:.1f} seconds")
        return results

    def _get_system_context(self, domain: str) -> str:
        """Get domain-specific system context"""
        contexts = {
            'defense': "You are a military AI assistant operating in a classified defense environment. Provide tactical analysis, threat assessment, and operational recommendations with high reliability and security considerations.",
            'aerospace': "You are an aerospace systems AI operating in mission-critical aviation environment. Provide engineering analysis, safety assessments, and operational decisions for aircraft and space systems.",
            'semiconductor': "You are a semiconductor process AI in a clean room manufacturing environment. Provide defect analysis, process optimization, and yield improvement recommendations for semiconductor fabrication.",
            'infrastructure': "You are a critical infrastructure AI monitoring national infrastructure systems. Provide structural analysis, risk assessment, and maintenance recommendations for power grids, transportation, and utilities."
        }
        return contexts.get(domain, "You are an AI assistant in a critical infrastructure environment.")

    def calculate_critical_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for critical infrastructure"""
        summary = {}

        # Group by model and domain
        model_domain_stats = {}
        model_overall_stats = {}
        domain_stats = {}

        for result in results:
            model = result['model']
            domain = result['domain']
            score = result['score']
            response_time = result['response_time']
            criticality = result['criticality']

            # Initialize stats
            if model not in model_domain_stats:
                model_domain_stats[model] = {}
                model_overall_stats[model] = {'scores': [], 'times': [], 'critical_scores': []}

            if domain not in model_domain_stats[model]:
                model_domain_stats[model][domain] = {'scores': [], 'times': []}

            if domain not in domain_stats:
                domain_stats[domain] = {'model_scores': {}}

            # Add data
            model_domain_stats[model][domain]['scores'].append(score)
            model_domain_stats[model][domain]['times'].append(response_time)
            model_overall_stats[model]['scores'].append(score)
            model_overall_stats[model]['times'].append(response_time)

            # Criticality-weighted scoring
            weight = {'critical': 2.0, 'high': 1.5, 'medium': 1.0, 'low': 0.5}[criticality]
            model_overall_stats[model]['critical_scores'].append(score * weight)

            # Domain stats
            if model not in domain_stats[domain]['model_scores']:
                domain_stats[domain]['model_scores'][model] = []
            domain_stats[domain]['model_scores'][model].append(score)

        # Calculate statistics
        for model, stats in model_overall_stats.items():
            scores = stats['scores']
            times = [t for t in stats['times'] if t > 0]
            critical_scores = stats['critical_scores']

            summary[f'{model}_overall_avg_score'] = np.mean(scores) if scores else 0
            summary[f'{model}_overall_score_std'] = np.std(scores) if len(scores) > 1 else 0
            summary[f'{model}_overall_avg_time'] = np.mean(times) if times else 0
            summary[f'{model}_overall_time_std'] = np.std(times) if len(times) > 1 else 0
            summary[f'{model}_critical_weighted_score'] = np.mean(critical_scores) if critical_scores else 0

            # Domain-specific stats
            for domain, domain_data in model_domain_stats[model].items():
                domain_scores = domain_data['scores']
                domain_times = [t for t in domain_data['times'] if t > 0]

                summary[f'{model}_{domain}_avg_score'] = np.mean(domain_scores) if domain_scores else 0
                summary[f'{model}_{domain}_score_std'] = np.std(domain_scores) if len(domain_scores) > 1 else 0
                summary[f'{model}_{domain}_avg_time'] = np.mean(domain_times) if domain_times else 0

        # Domain comparison stats
        for domain, stats in domain_stats.items():
            summary[f'{domain}_avg_score'] = np.mean([score for scores in stats['model_scores'].values() for score in scores])

            # Best performer per domain
            domain_performers = {}
            for model, scores in stats['model_scores'].items():
                domain_performers[model] = np.mean(scores) if scores else 0

            best_model = max(domain_performers.items(), key=lambda x: x[1])
            summary[f'{domain}_best_model'] = best_model[0]
            summary[f'{domain}_best_score'] = best_model[1]

        return summary

    def create_critical_visualizations(self, results: Dict[str, Any]) -> List[str]:
        """Create comprehensive visualizations for critical infrastructure"""
        plt.style.use('default')
        sns.set_palette("husl")

        charts = []

        # 1. Overall Critical Infrastructure Performance
        charts.extend(self._create_overall_critical_chart(results))

        # 2. Domain-specific Performance
        charts.extend(self._create_domain_performance_charts(results))

        # 3. Criticality-weighted Performance
        charts.extend(self._create_criticality_analysis(results))

        # 4. Response Time Reliability Analysis
        charts.extend(self._create_reliability_analysis(results))

        return charts

    def _create_overall_critical_chart(self, results: Dict[str, Any]) -> List[str]:
        """Create overall critical infrastructure performance chart"""
        charts = []
        summary = results['summary']
        models = list(self.models.keys())

        # Overall performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scores = [summary.get(f'{model}_overall_avg_score', 0) for model in models]
        score_stds = [summary.get(f'{model}_overall_score_std', 0) for model in models]

        bars1 = ax1.bar(models, scores, yerr=score_stds, capsize=5, alpha=0.8)
        ax1.set_title('Critical Infrastructure Overall Performance')
        ax1.set_ylabel('Average Score')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3)

        # Criticality-weighted scores
        critical_scores = [summary.get(f'{model}_critical_weighted_score', 0) for model in models]

        bars2 = ax2.bar(models, critical_scores, alpha=0.8, color='orange')
        ax2.set_title('Criticality-Weighted Performance\n(Higher weight for critical scenarios)')
        ax2.set_ylabel('Weighted Score')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bars, ax in [(bars1, ax1), (bars2, ax2)]:
            for bar, score in zip(bars, scores if ax == ax1 else critical_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_critical_infrastructure_overall.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        return charts

    def _create_domain_performance_charts(self, results: Dict[str, Any]) -> List[str]:
        """Create domain-specific performance charts"""
        charts = []
        summary = results['summary']
        models = list(self.models.keys())
        domains = list(self.benchmarks.keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        domain_titles = {
            'defense': 'Defense & Military',
            'aerospace': 'Aerospace & Aviation',
            'semiconductor': 'Semiconductor Manufacturing',
            'infrastructure': 'Critical Infrastructure'
        }

        for idx, domain in enumerate(domains):
            ax = axes[idx]

            scores = [summary.get(f'{model}_{domain}_avg_score', 0) for model in models]
            score_stds = [summary.get(f'{model}_{domain}_score_std', 0) for model in models]

            bars = ax.bar(models, scores, yerr=score_stds, capsize=5, alpha=0.8)
            ax.set_title(f'{domain_titles[domain]}\nPerformance by Model')
            ax.set_ylabel('Average Score')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_critical_domains_detailed.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        return charts

    def _create_criticality_analysis(self, results: Dict[str, Any]) -> List[str]:
        """Create criticality analysis chart"""
        charts = []

        # Group results by criticality
        criticality_data = {'critical': [], 'high': [], 'medium': []}

        for result in results['results']:
            crit = result['criticality']
            if crit in criticality_data:
                criticality_data[crit].append(result['score'])

        fig, ax = plt.subplots(figsize=(12, 6))

        criticality_labels = {'critical': 'Critical (2x weight)', 'high': 'High (1.5x weight)', 'medium': 'Medium (1x weight)'}
        x_pos = range(len(criticality_data))

        for i, (crit, scores) in enumerate(criticality_data.items()):
            if scores:
                bp = ax.boxplot([scores], positions=[i], widths=0.6,
                               patch_artist=True, showmeans=True, meanline=True)
                bp['boxes'][0].set_facecolor(['red', 'orange', 'green'][i])
                bp['means'][0].set_color('black')
                bp['means'][0].set_linewidth(2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([criticality_labels[c] for c in criticality_data.keys()])
        ax.set_title('Performance Distribution by Scenario Criticality')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_criticality_analysis.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        charts.append(str(chart_file))

        return charts

    def _create_reliability_analysis(self, results: Dict[str, Any]) -> List[str]:
        """Create response time reliability analysis"""
        charts = []

        # Extract response times by model
        model_times = {}
        for result in results['results']:
            model = result['model']
            time = result['response_time']
            if time > 0:  # Exclude timeouts
                if model not in model_times:
                    model_times[model] = []
                model_times[model].append(time)

        if model_times:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Create box plot for response times
            time_data = [model_times[model] for model in self.models.keys() if model in model_times]
            labels = [model for model in self.models.keys() if model in model_times]

            bp = ax.boxplot(time_data, labels=labels, patch_artist=True, showmeans=True, meanline=True)

            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            bp['means'][0].set_color('red')
            bp['means'][0].set_linewidth(2)

            ax.set_title('Response Time Reliability Analysis\n(Critical Infrastructure Scenarios)')
            ax.set_ylabel('Response Time (seconds)')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = ""
            for i, model in enumerate(labels):
                if model in model_times and model_times[model]:
                    times = model_times[model]
                    stats_text += f"{model}: μ={np.mean(times):.1f}s, σ={np.std(times):.1f}s\n"

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            chart_file = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_response_time_reliability.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(str(chart_file))

        return charts

    def save_critical_results(self, results: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Save critical infrastructure benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_file = self.output_dir / f"{timestamp}_critical_infrastructure_abc_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Create visualizations
        chart_files = self.create_critical_visualizations(results)

        # Create comprehensive report
        report_file = self.output_dir / f"{timestamp}_critical_infrastructure_abc_report.md"
        self._create_critical_report(results, report_file, chart_files)

        return str(json_file), str(report_file), chart_files

    def _create_critical_report(self, results: Dict[str, Any], report_file: Path, chart_files: List[str]):
        """Create comprehensive critical infrastructure report"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Critical Infrastructure ABC Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Test Environment:** Closed Domain, On-Premises\n")
            f.write(f"**Total Execution Time:** {results['metadata']['total_execution_time']:.1f} seconds\n\n")

            f.write("## Executive Summary\n\n")
            self._write_critical_executive_summary(f, results)

            f.write("## Domain-Specific Analysis\n\n")
            self._write_domain_analysis(f, results)

            f.write("## Criticality Assessment\n\n")
            self._write_criticality_assessment(f, results)

            f.write("## Performance Metrics\n\n")
            self._write_critical_metrics(f, results)

            f.write("## Visualizations\n\n")
            for chart in chart_files:
                chart_name = Path(chart).name
                f.write(f"![{chart_name}]({chart_name})\n\n")

            f.write("## Recommendations for Critical Infrastructure\n\n")
            self._write_critical_recommendations(f, results)

def main():
    """Main execution function"""
    print("[START] Critical Infrastructure ABC Benchmark")
    print("=" * 80)

    benchmark = CriticalInfrastructureBenchmark()
    results = benchmark.run_critical_benchmark()

    # Save results and create report
    json_file, report_file, chart_files = benchmark.save_critical_results(results)

    print("\n[RESULTS]")
    print(f"  JSON: {json_file}")
    print(f"  Report: {report_file}")
    print(f"  Charts: {len(chart_files)} files")

    # Print summary
    summary = results['summary']
    models = list(benchmark.models.keys())

    print("\n[CRITICAL INFRASTRUCTURE PERFORMANCE]")
    print("=" * 60)
    for model in models:
        score = summary.get(f'{model}_overall_avg_score', 0)
        time = summary.get(f'{model}_overall_avg_time', 0)
        critical_score = summary.get(f'{model}_critical_weighted_score', 0)
        print(f"  {model}: Score={score:.3f}, Time={time:.1f}s, Critical={critical_score:.3f}")
    # Domain winners
    domains = list(benchmark.benchmarks.keys())
    print("\n[DOMAIN LEADERS]")
    for domain in domains:
        best_model = summary.get(f'{domain}_best_model', 'N/A')
        best_score = summary.get(f'{domain}_best_score', 0)
        print(f"  {domain}: {best_model} ({best_score:.3f})")
    # Play completion sound
    try:
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/utils/play_audio_notification.ps1"
        ])
    except:
        pass

    print("\n✅ Critical Infrastructure ABC Benchmark completed successfully!")

if __name__ == "__main__":
    main()
