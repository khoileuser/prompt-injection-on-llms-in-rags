import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

from src.detection import DetectionResult, AttackDetectionResult
from src.inference import InferenceResult
from src.config_loader import ModelConfig, AttackVariation


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """
    Complete result of a single attack test.

    Attributes:
        timestamp: When the test was run
        model_key: Model identifier
        model_name: Human-readable model name
        model_parameters: Model size in billions
        attack_id: Attack identifier
        attack_name: Human-readable attack name
        attack_category: Attack category (e.g., 'instruction_override')
        defense_strategy: Defense prompt strategy used
        prompt: The attack prompt used
        response: Model's response
        detection_result: SUCCESS/BLOCKED/ERROR
        tokens_generated: Number of output tokens
        response_time: Inference time in seconds
        explanation: Human-readable detection explanation
        injection_vector: Dimension 1 (direct/indirect)
        attack_objective: Dimension 2 (instruction_override, etc.)
    """
    timestamp: str
    model_key: str
    model_name: str
    model_parameters: float
    attack_id: str
    attack_name: str
    attack_category: str
    defense_strategy: str
    prompt: str
    response: str
    detection_result: str
    tokens_generated: int
    response_time: float
    explanation: str
    injection_vector: str = "direct"
    attack_objective: str = "instruction_override"


@dataclass
class ModelMetrics:
    """
    Aggregated metrics for a single model.

    Attributes:
        model_key: Model identifier
        model_name: Human-readable model name
        model_parameters: Model size in billions
        total_tests: Number of tests run
        successful_attacks: Number of successful attacks
        blocked_attacks: Number of blocked attacks
        error_count: Number of errors
        asr: Attack Success Rate (successful / total)
        avg_response_time: Average inference time
        avg_tokens: Average tokens generated
        category_asr: ASR broken down by attack category
    """
    model_key: str
    model_name: str
    model_parameters: float
    total_tests: int = 0
    successful_attacks: int = 0
    blocked_attacks: int = 0
    error_count: int = 0
    asr: float = 0.0
    avg_response_time: float = 0.0
    avg_tokens: float = 0.0
    category_asr: Dict[str, float] = field(default_factory=dict)


@dataclass
class AttackMetrics:
    """
    Aggregated metrics for a single attack type.

    Attributes:
        attack_category: Category name
        total_tests: Number of tests across all models
        successful_attacks: Number of successful attacks
        effectiveness: Overall effectiveness rate
        model_success: Success rate per model
    """
    attack_category: str
    total_tests: int = 0
    successful_attacks: int = 0
    effectiveness: float = 0.0
    model_success: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and stores metrics from prompt injection tests.

    This class provides:
    - Result storage and retrieval
    - Aggregation by model and attack type
    - CSV and JSON export capabilities
    - Real-time metrics calculation
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the metrics collector.

        Args:
            output_dir: Directory for saving results (default: ./results)
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent / "results"
        else:
            self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for test results
        self.results: List[TestResult] = []

        # Session metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()

        logger.info(
            f"MetricsCollector initialized. Session: {self.session_id}")

    def add_result(
        self,
        model_config: ModelConfig,
        attack: AttackVariation,
        attack_category: str,
        inference_result: InferenceResult,
        detection_result: AttackDetectionResult,
        defense_strategy: str = None
    ) -> TestResult:
        """
        Add a single test result to the collection.

        Args:
            model_config: Configuration of the tested model
            attack: The attack variation that was tested
            attack_category: Category of the attack
            inference_result: Result from the inference engine
            detection_result: Result from the detector
            defense_strategy: The defense strategy used (auto-detected if None)

        Returns:
            The created TestResult object
        """
        # Auto-detect defense strategy if not provided
        if defense_strategy is None:
            from src.defense_prompts import get_defense_strategy
            defense_strategy = get_defense_strategy().value

        # Extract dimensions from attack
        injection_vector = getattr(attack, 'injection_vector', 'direct')
        attack_objective = getattr(attack, 'attack_objective', attack_category)

        result = TestResult(
            timestamp=datetime.now().isoformat(),
            model_key=model_config.key,
            model_name=model_config.name,
            model_parameters=model_config.parameters,
            attack_id=attack.id,
            attack_name=attack.name,
            attack_category=attack_category,
            defense_strategy=defense_strategy,
            prompt=attack.prompt,
            response=inference_result.response,
            detection_result=detection_result.result.value,
            tokens_generated=inference_result.tokens_generated,
            response_time=inference_result.response_time,
            # Changed from explanation to reasoning
            explanation=detection_result.reasoning,
            injection_vector=injection_vector,
            attack_objective=attack_objective
        )

        self.results.append(result)
        logger.debug(
            f"Added result: {model_config.key}/{attack.id} -> {detection_result.result.value}")

        return result

    def get_model_metrics(self) -> Dict[str, ModelMetrics]:
        """
        Calculate aggregated metrics for each model.

        Returns:
            Dictionary mapping model_key to ModelMetrics
        """
        metrics: Dict[str, ModelMetrics] = {}

        # Group results by model
        model_results: Dict[str, List[TestResult]] = defaultdict(list)
        for result in self.results:
            model_results[result.model_key].append(result)

        # Calculate metrics for each model
        for model_key, results in model_results.items():
            if not results:
                continue

            # Basic counts
            total = len(results)
            successful = sum(
                1 for r in results if r.detection_result == "success")
            blocked = sum(
                1 for r in results if r.detection_result == "blocked")
            errors = sum(1 for r in results if r.detection_result == "error")

            # Calculate category-specific ASR
            category_results: Dict[str, List[TestResult]] = defaultdict(list)
            for r in results:
                category_results[r.attack_category].append(r)

            category_asr = {}
            for category, cat_results in category_results.items():
                cat_successful = sum(
                    1 for r in cat_results if r.detection_result == "success")
                category_asr[category] = cat_successful / \
                    len(cat_results) if cat_results else 0.0

            # Average metrics
            avg_time = sum(r.response_time for r in results) / total
            avg_tokens = sum(r.tokens_generated for r in results) / total

            metrics[model_key] = ModelMetrics(
                model_key=model_key,
                model_name=results[0].model_name,
                model_parameters=results[0].model_parameters,
                total_tests=total,
                successful_attacks=successful,
                blocked_attacks=blocked,
                error_count=errors,
                asr=successful / total if total > 0 else 0.0,
                avg_response_time=avg_time,
                avg_tokens=avg_tokens,
                category_asr=category_asr
            )

        return metrics

    def get_attack_metrics(self) -> Dict[str, AttackMetrics]:
        """
        Calculate aggregated metrics for each attack category.

        Returns:
            Dictionary mapping attack_category to AttackMetrics
        """
        metrics: Dict[str, AttackMetrics] = {}

        # Group results by attack category
        category_results: Dict[str, List[TestResult]] = defaultdict(list)
        for result in self.results:
            category_results[result.attack_category].append(result)

        # Calculate metrics for each category
        for category, results in category_results.items():
            if not results:
                continue

            total = len(results)
            successful = sum(
                1 for r in results if r.detection_result == "success")

            # Per-model success rate
            model_results: Dict[str, List[TestResult]] = defaultdict(list)
            for r in results:
                model_results[r.model_key].append(r)

            model_success = {}
            for model_key, model_res in model_results.items():
                model_successful = sum(
                    1 for r in model_res if r.detection_result == "success")
                model_success[model_key] = model_successful / \
                    len(model_res) if model_res else 0.0

            metrics[category] = AttackMetrics(
                attack_category=category,
                total_tests=total,
                successful_attacks=successful,
                effectiveness=successful / total if total > 0 else 0.0,
                model_success=model_success
            )

        return metrics

    def get_variation_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics for individual attack variations.

        Returns:
            Dictionary mapping attack_id to variation metrics
        """
        variation_results: Dict[str, List[TestResult]] = defaultdict(list)
        for result in self.results:
            variation_results[result.attack_id].append(result)

        metrics = {}
        for attack_id, results in variation_results.items():
            if not results:
                continue

            total = len(results)
            successful = sum(
                1 for r in results if r.detection_result == "success")

            metrics[attack_id] = {
                "attack_id": attack_id,
                "attack_name": results[0].attack_name,
                "attack_category": results[0].attack_category,
                "total_tests": total,
                "successful_attacks": successful,
                "success_rate": successful / total if total > 0 else 0.0,
            }

        return metrics

    def get_heatmap_data(self) -> Tuple[List[str], List[str], List[List[float]]]:
        """
        Generate data for a model x attack heatmap.

        Returns:
            Tuple of (model_names, attack_categories, asr_matrix)
            where asr_matrix[i][j] is the ASR for model i on attack category j
        """
        model_metrics = self.get_model_metrics()

        # Get unique models and categories
        models = list(model_metrics.keys())
        categories = set()
        for mm in model_metrics.values():
            categories.update(mm.category_asr.keys())
        categories = sorted(list(categories))

        # Build matrix
        matrix = []
        model_names = []
        for model_key in models:
            mm = model_metrics[model_key]
            model_names.append(mm.model_name)
            row = [mm.category_asr.get(cat, 0.0) for cat in categories]
            matrix.append(row)

        return model_names, categories, matrix

    def get_trend_data(self) -> Dict[str, List[float]]:
        """
        Calculate ASR trends across attack variations per category.

        Returns:
            Dictionary mapping category to list of cumulative ASR values
        """
        # Group results by category and variation order
        category_variations: Dict[str, Dict[str, List[TestResult]]] = defaultdict(
            lambda: defaultdict(list))

        for result in self.results:
            category_variations[result.attack_category][result.attack_id].append(
                result)

        trends = {}
        for category, variations in category_variations.items():
            # Sort variations by ID to maintain order
            sorted_vars = sorted(variations.items(), key=lambda x: x[0])

            cumulative_asr = []
            total_tests = 0
            total_success = 0

            for var_id, results in sorted_vars:
                total_tests += len(results)
                total_success += sum(1 for r in results if r.detection_result == "success")
                cumulative_asr.append(
                    total_success / total_tests if total_tests > 0 else 0.0)

            trends[category] = cumulative_asr

        return trends

    def get_radar_data(self) -> Dict[str, Dict[str, float]]:
        """
        Generate radar/spider chart data for model robustness.

        Returns:
            Dictionary mapping model_key to dict of (category -> robustness_score)
            Robustness = 1 - ASR (higher is better)
        """
        model_metrics = self.get_model_metrics()

        radar_data = {}
        for model_key, mm in model_metrics.items():
            radar_data[model_key] = {
                cat: 1.0 - asr for cat, asr in mm.category_asr.items()
            }

        return radar_data

    #  METRICS (2×3 Matrix)
    # These methods support the lightweight framework that decomposes
    # attacks by injection vector (direct/indirect) and attack objective
    # (instruction_override, data_extraction, role_confusion).

    def get_taxonomy_matrix_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Calculate ASR for each cell in the 2×3 matrix.

        Returns:
            Nested dict: {injection_vector: {attack_objective: {metrics}}}
            Each cell contains: total, successful, blocked, asr
        """
        # Initialize matrix structure
        vectors = ['direct', 'indirect']
        objectives = ['instruction_override',
                      'data_extraction', 'role_confusion']

        matrix = {}
        for vector in vectors:
            matrix[vector] = {}
            for objective in objectives:
                matrix[vector][objective] = {
                    'total': 0,
                    'successful': 0,
                    'blocked': 0,
                    'asr': 0.0
                }

        # Populate from results
        for result in self.results:
            vector = result.injection_vector
            objective = result.attack_objective

            if vector in matrix and objective in matrix.get(vector, {}):
                cell = matrix[vector][objective]
                cell['total'] += 1
                if result.detection_result == "success":
                    cell['successful'] += 1
                else:
                    cell['blocked'] += 1

        # Calculate ASR for each cell
        for vector in vectors:
            for objective in objectives:
                cell = matrix[vector][objective]
                if cell['total'] > 0:
                    cell['asr'] = cell['successful'] / cell['total']

        return matrix

    def get_taxonomy_asr_table(self) -> List[Dict[str, Any]]:
        """
        Get matrix as a flat table for display.

        Returns:
            List of dicts with vector, objective, total, successful, asr
        """
        matrix = self.get_taxonomy_matrix_data()
        table = []

        for vector, objectives in matrix.items():
            for objective, metrics in objectives.items():
                table.append({
                    'injection_vector': vector.replace('_', ' ').title(),
                    'attack_objective': objective.replace('_', ' ').title(),
                    'total_tests': metrics['total'],
                    'successful_attacks': metrics['successful'],
                    'blocked_attacks': metrics['blocked'],
                    'asr': metrics['asr'],
                    'asr_percent': f"{metrics['asr'] * 100:.1f}%"
                })

        return table

    def get_taxonomy_heatmap_data(self) -> Tuple[List[str], List[str], List[List[float]]]:
        """
        Generate data for matrix heatmap visualization.

        Returns:
            Tuple of (vector_labels, objective_labels, asr_matrix)
        """
        matrix = self.get_taxonomy_matrix_data()

        vectors = ['Direct', 'Indirect']
        objectives = ['Instruction Override',
                      'Data Extraction', 'Role Confusion']

        vector_keys = ['direct', 'indirect']
        objective_keys = ['instruction_override',
                          'data_extraction', 'role_confusion']

        asr_matrix = []
        for v_key in vector_keys:
            row = []
            for o_key in objective_keys:
                row.append(matrix[v_key][o_key]['asr'])
            asr_matrix.append(row)

        return vectors, objectives, asr_matrix

    def get_vector_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Compare ASR between direct and indirect injection vectors.

        Returns:
            Dict with 'direct' and 'indirect' keys, each containing per-objective ASR
        """
        matrix = self.get_taxonomy_matrix_data()

        comparison = {}
        for vector in ['direct', 'indirect']:
            total = sum(m['total'] for m in matrix[vector].values())
            successful = sum(m['successful'] for m in matrix[vector].values())
            comparison[vector] = {
                'overall_asr': successful / total if total > 0 else 0.0,
                'total_tests': total,
                'objectives': {
                    obj: data['asr']
                    for obj, data in matrix[vector].items()
                }
            }

        return comparison

    def get_objective_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Compare ASR across different attack objectives.

        Returns:
            Dict with objective keys, each containing per-vector ASR
        """
        matrix = self.get_taxonomy_matrix_data()
        objectives = ['instruction_override',
                      'data_extraction', 'role_confusion']

        comparison = {}
        for objective in objectives:
            total = sum(matrix[v][objective]['total']
                        for v in ['direct', 'indirect'])
            successful = sum(matrix[v][objective]['successful']
                             for v in ['direct', 'indirect'])
            comparison[objective] = {
                'overall_asr': successful / total if total > 0 else 0.0,
                'total_tests': total,
                'vectors': {
                    v: matrix[v][objective]['asr']
                    for v in ['direct', 'indirect']
                }
            }

        return comparison

    def calculate_defense_effectiveness(
        self,
        baseline_results: List[TestResult],
        defense_name: str = "unknown"
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Calculate Defense Effectiveness (DE) comparing current results to baseline.

        DE = 1 - (ASR_defended / ASR_baseline)

        Args:
            baseline_results: Results from baseline (no defense) testing
            defense_name: Name of the defense being evaluated

        Returns:
            Nested dict: {injection_vector: {attack_objective: DE_value}}
        """
        # Calculate baseline matrix
        baseline_matrix = self._calculate_matrix_from_results(baseline_results)

        # Calculate defended matrix (current results)
        defended_matrix = self.get_taxonomy_matrix_data()

        # Calculate DE for each cell
        vectors = ['direct', 'indirect']
        objectives = ['instruction_override',
                      'data_extraction', 'role_confusion']

        de_matrix = {}
        for vector in vectors:
            de_matrix[vector] = {}
            for objective in objectives:
                baseline_asr = baseline_matrix[vector][objective]['asr']
                defended_asr = defended_matrix[vector][objective]['asr']

                if baseline_asr == 0:
                    # Can't calculate DE if baseline has 0% ASR
                    de_matrix[vector][objective] = None
                else:
                    de_matrix[vector][objective] = 1 - \
                        (defended_asr / baseline_asr)

        return de_matrix

    def _calculate_matrix_from_results(self, results: List[TestResult]) -> Dict:
        """Helper to calculate matrix from a list of results."""
        vectors = ['direct', 'indirect']
        objectives = ['instruction_override',
                      'data_extraction', 'role_confusion']

        matrix = {}
        for vector in vectors:
            matrix[vector] = {}
            for objective in objectives:
                matrix[vector][objective] = {
                    'total': 0,
                    'successful': 0,
                    'asr': 0.0
                }

        for result in results:
            vector = result.injection_vector
            objective = result.attack_objective

            if vector in matrix and objective in matrix.get(vector, {}):
                cell = matrix[vector][objective]
                cell['total'] += 1
                if result.detection_result == "success":
                    cell['successful'] += 1

        for vector in vectors:
            for objective in objectives:
                cell = matrix[vector][objective]
                if cell['total'] > 0:
                    cell['asr'] = cell['successful'] / cell['total']

        return matrix

    def get_taxonomy_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary using the matrix framework.

        Returns:
            Dictionary containing matrix-based analysis
        """
        matrix = self.get_taxonomy_matrix_data()
        vector_comparison = self.get_vector_comparison()
        objective_comparison = self.get_objective_comparison()

        # Find most/least vulnerable cells
        all_cells = []
        for vector, objectives in matrix.items():
            for objective, metrics in objectives.items():
                if metrics['total'] > 0:
                    all_cells.append({
                        'vector': vector,
                        'objective': objective,
                        'asr': metrics['asr'],
                        'total': metrics['total']
                    })

        sorted_cells = sorted(all_cells, key=lambda x: x['asr'], reverse=True)

        return {
            'matrix': matrix,
            'vector_comparison': vector_comparison,
            'objective_comparison': objective_comparison,
            'most_vulnerable_cell': sorted_cells[0] if sorted_cells else None,
            'least_vulnerable_cell': sorted_cells[-1] if sorted_cells else None,
            'total_tests': sum(c['total'] for c in all_cells),
            'overall_asr': (
                sum(c['asr'] * c['total'] for c in all_cells) /
                sum(c['total'] for c in all_cells)
            ) if all_cells else 0.0
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all collected metrics.

        Returns:
            Dictionary containing summary statistics
        """
        model_metrics = self.get_model_metrics()
        attack_metrics = self.get_attack_metrics()

        if not self.results:
            return {"error": "No results collected"}

        # Count unique defense strategies
        unique_defenses = len(set(r.defense_strategy for r in self.results))

        # Find best/worst performing
        models_by_asr = sorted(model_metrics.values(), key=lambda m: m.asr)
        attacks_by_effectiveness = sorted(
            attack_metrics.values(), key=lambda a: a.effectiveness, reverse=True)

        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "total_tests": len(self.results),
            "unique_models": len(model_metrics),
            "unique_attack_categories": len(attack_metrics),
            "unique_defense_strategies": unique_defenses,
            "overall_asr": sum(m.asr for m in model_metrics.values()) / len(model_metrics) if model_metrics else 0.0,
            "most_robust_model": models_by_asr[0].model_name if models_by_asr else "N/A",
            "most_vulnerable_model": models_by_asr[-1].model_name if models_by_asr else "N/A",
            "most_effective_attack": attacks_by_effectiveness[0].attack_category if attacks_by_effectiveness else "N/A",
            "least_effective_attack": attacks_by_effectiveness[-1].attack_category if attacks_by_effectiveness else "N/A",
        }

    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export all results to a CSV file.

        Args:
            filename: Output filename (default: results_{session_id}.csv)

        Returns:
            Path to the created CSV file
        """
        if filename is None:
            filename = f"results_{self.session_id}.csv"

        filepath = self.output_dir / filename

        # Define CSV columns (including defense_strategy)
        fieldnames = [
            "timestamp", "model_key", "model_name", "model_parameters",
            "attack_id", "attack_name", "attack_category",
            "defense_strategy",
            "detection_result", "tokens_generated", "response_time",
            "prompt", "response", "explanation"
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for result in self.results:
                row = asdict(result)
                writer.writerow(row)

        logger.info(f"Exported {len(self.results)} results to {filepath}")
        return str(filepath)

    def export_to_json(self, filename: Optional[str] = None) -> str:
        """
        Export all results and metrics to a JSON file.

        Args:
            filename: Output filename (default: results_{session_id}.json)

        Returns:
            Path to the created JSON file
        """
        if filename is None:
            filename = f"results_{self.session_id}.json"

        filepath = self.output_dir / filename

        export_data = {
            "summary": self.get_summary(),
            "model_metrics": {k: asdict(v) for k, v in self.get_model_metrics().items()},
            "attack_metrics": {k: asdict(v) for k, v in self.get_attack_metrics().items()},
            "variation_metrics": self.get_variation_metrics(),
            "results": [asdict(r) for r in self.results]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported metrics to {filepath}")
        return str(filepath)

    def load_from_file(self, filepath: str) -> int:
        """
        Load results from a previously exported JSON file.

        Args:
            filepath: Path to the JSON file to load

        Returns:
            Number of results loaded

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load results from the 'results' key
        if 'results' not in data:
            logger.warning(f"No 'results' key found in {filepath}")
            return 0

        # Clear existing results and load new ones
        self.results.clear()

        for result_dict in data['results']:
            # Convert dict back to TestResult
            result = TestResult(**result_dict)
            self.results.append(result)

        # Update session metadata
        if 'summary' in data and 'session_id' in data['summary']:
            self.session_id = data['summary']['session_id']

        logger.info(f"Loaded {len(self.results)} results from {filepath}")
        return len(self.results)

    def load_from_csv(self, filepath: str) -> int:
        """
        Load results from a previously exported CSV file.

        Args:
            filepath: Path to the CSV file to load

        Returns:
            Number of results loaded

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        import pandas as pd

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        # Read CSV file
        df = pd.read_csv(filepath)

        if df.empty:
            logger.warning(f"No data found in {filepath}")
            return 0

        # Clear existing results and load new ones
        self.results.clear()

        for _, row in df.iterrows():
            # Convert CSV row to TestResult
            result = TestResult(
                timestamp=row['timestamp'],
                model_key=row['model_key'],
                model_name=row['model_name'],
                model_parameters=float(row['model_parameters']),
                attack_id=row['attack_id'],
                attack_name=row['attack_name'],
                attack_category=row['attack_category'],
                defense_strategy=row['defense_strategy'],
                detection_result=DetectionResult[row['detection_result'].upper(
                )],
                tokens_generated=int(row['tokens_generated']),
                response_time=float(row['response_time']),
                prompt=row['prompt'],
                response=row['response'],
                explanation=row['explanation']
            )
            self.results.append(result)

        # Extract session_id from filename (e.g., attack_results_20251231_120135.csv)
        filename = filepath.stem  # attack_results_20251231_120135
        parts = filename.split('_')
        if len(parts) >= 3:
            self.session_id = '_'.join(parts[-2:])  # 20251231_120135

        logger.info(f"Loaded {len(self.results)} results from {filepath}")
        return len(self.results)

    def clear(self):
        """Clear all collected results."""
        self.results.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        logger.info("Metrics cleared, new session started")


_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global MetricsCollector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
