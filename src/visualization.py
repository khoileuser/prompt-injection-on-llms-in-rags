from src.metrics import MetricsCollector, get_metrics_collector
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import io
import base64
import warnings
from typing import Dict, Optional, Any
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web apps


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Visualizer:
    """
    Creates visualizations from metrics data.

    All methods return either a matplotlib Figure or a base64-encoded
    PNG image string suitable for embedding in web pages.
    """

    def __init__(self, collector: Optional[MetricsCollector] = None):
        """
        Initialize the visualizer.

        Args:
            collector: MetricsCollector instance (uses global if not provided)
        """
        self.collector = collector or get_metrics_collector()

        # Color scheme for consistency
        self.colors = {
            'primary': '#2563eb',
            'success': '#16a34a',
            'warning': '#ea580c',
            'danger': '#dc2626',
            'neutral': '#6b7280',
            'models': ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#84cc16'],
            'attacks': ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4'],
        }

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64-encoded PNG.

        Args:
            fig: matplotlib Figure object

        Returns:
            Base64-encoded PNG string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def create_asr_bar_chart(
        self,
        title: str = "Attack Success Rate by Model",
        as_base64: bool = True
    ) -> Any:
        """
        Create a bar chart comparing ASR across models.

        This chart shows:
        - Overall ASR for each model
        - Color coding by vulnerability level

        Args:
            title: Chart title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        model_metrics = self.collector.get_model_metrics()

        if not model_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Prepare data
        models = list(model_metrics.values())
        names = [m.model_name for m in models]
        asrs = [m.asr * 100 for m in models]  # Convert to percentage

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Color bars by vulnerability level
        colors = []
        for asr in asrs:
            if asr < 20:
                colors.append(self.colors['success'])
            elif asr < 40:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['danger'])

        # Create bars
        bars = ax.bar(names, asrs, color=colors,
                      edgecolor='white', linewidth=1.5)

        # Add value labels on bars
        for bar, asr in zip(bars, asrs):
            height = bar.get_height()
            ax.annotate(f'{asr:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

        # Styling
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)',
                      fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, max(asrs) * 1.2 if asrs else 100)

        # Add legend
        legend_elements = [
            Patch(facecolor=self.colors['success'], label='Low Risk (<20%)'),
            Patch(facecolor=self.colors['warning'],
                  label='Medium Risk (20-40%)'),
            Patch(facecolor=self.colors['danger'], label='High Risk (>40%)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Rotate x labels if needed
        if len(names) > 4:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_heatmap(
        self,
        title: str = "Attack Success Rate: Model × Attack Category",
        as_base64: bool = True
    ) -> Any:
        """
        Create a heatmap showing ASR for each model-attack combination.

        This visualization helps identify:
        - Which attack types are most effective against which models
        - Overall model vulnerabilities
        - Attack type effectiveness patterns

        Args:
            title: Chart title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        model_names, categories, matrix = self.collector.get_heatmap_data()

        if not matrix or not model_names or not categories:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Convert to numpy array and to percentage
        data = np.array(matrix) * 100

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create heatmap with custom colormap (green=low ASR, red=high ASR)
        cmap = sns.diverging_palette(145, 10, s=80, l=55, as_cmap=True)

        # Shorten category names for display
        short_categories = [c.replace('_', ' ').title() for c in categories]

        sns.heatmap(
            data,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            xticklabels=short_categories,
            yticklabels=model_names,
            cbar_kws={'label': 'Attack Success Rate (%)'},
            linewidths=0.5,
            linecolor='white',
            ax=ax
        )

        # Styling
        ax.set_xlabel('Attack Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Rotate x labels
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_trend_line_chart(
        self,
        title: str = "ASR Trend Across Attack Variations",
        as_base64: bool = True
    ) -> Any:
        """
        Create a line chart showing ASR trends across variations per category.

        This chart shows:
        - How ASR evolves as more attack variations are tested
        - Different trends for each attack category
        - Cumulative effectiveness patterns

        Args:
            title: Chart title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        trends = self.collector.get_trend_data()

        if not trends:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each category
        colors = self.colors['attacks']
        markers = ['o', 's', '^', 'D', 'v']

        for idx, (category, values) in enumerate(sorted(trends.items())):
            x = list(range(1, len(values) + 1))
            y = [v * 100 for v in values]  # Convert to percentage

            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            label = category.replace('_', ' ').title()

            ax.plot(x, y, marker=marker, markersize=8, linewidth=2,
                    color=color, label=label)

        # Styling
        ax.set_xlabel('Number of Variations Tested',
                      fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative ASR (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_radar_chart(
        self,
        title: str = "Model Robustness Profile",
        as_base64: bool = True
    ) -> Any:
        """
        Create a radar/spider chart showing model robustness across attack types.

        Robustness is calculated as 1 - ASR, so higher values mean
        better defense against that attack type.

        Args:
            title: Chart title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        radar_data = self.collector.get_radar_data()

        if not radar_data:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Get all categories
        all_categories = set()
        for model_data in radar_data.values():
            all_categories.update(model_data.keys())
        categories = sorted(list(all_categories))

        if not categories:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, 'No categories available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Number of categories
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Plot each model
        colors = self.colors['models']
        model_metrics = self.collector.get_model_metrics()

        for idx, (model_key, robustness) in enumerate(radar_data.items()):
            # Convert to percentage
            values = [robustness.get(cat, 0.5) * 100 for cat in categories]
            values += values[:1]  # Complete the loop

            color = colors[idx % len(colors)]
            model_name = model_metrics[model_key].model_name if model_key in model_metrics else model_key

            ax.plot(angles, values, 'o-', linewidth=2,
                    color=color, label=model_name)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Set category labels
        short_categories = [c.replace('_', ' ').title() for c in categories]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_categories, fontsize=10)

        # Set y limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])

        # Title and legend
        ax.set_title(title + '\n(Higher = More Robust)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_defense_comparison_chart(
        self,
        title: str = "Defense Strategy Effectiveness Comparison",
        as_base64: bool = True
    ) -> Any:
        """
        Create a grouped bar chart comparing ASR across different defense strategies by model.

        This chart shows:
        - ASR for each defense strategy broken down by model
        - Baseline (no defense) for comparison
        - Percentage reduction from baseline

        Args:
            title: Chart title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        from src.defense_prompts import DEFENSE_CONFIGS, DefenseStrategy
        import pandas as pd

        results = self.collector.results

        if not results:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available\nRun comparative tests first!',
                    ha='center', va='center', fontsize=14, style='italic')
            ax.axis('off')
            return self._fig_to_base64(fig) if as_base64 else fig

        # Group by defense strategy AND model
        from collections import defaultdict
        defense_model_stats = defaultdict(
            lambda: defaultdict(lambda: {'total': 0, 'success': 0}))

        for result in results:
            strategy = result.defense_strategy
            model = result.model_name
            defense_model_stats[strategy][model]['total'] += 1
            if result.detection_result == 'success':
                defense_model_stats[strategy][model]['success'] += 1

        if not defense_model_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No defense strategy data available',
                    ha='center', va='center', fontsize=14, style='italic')
            ax.axis('off')
            return self._fig_to_base64(fig) if as_base64 else fig

        # Prepare data for grouped bar chart
        strategies = sorted(defense_model_stats.keys())
        models = sorted(set(model for stats in defense_model_stats.values()
                        for model in stats.keys()))

        # Get proper strategy labels
        strategy_labels = []
        for strategy_key in strategies:
            try:
                strategy_enum = DefenseStrategy(strategy_key)
                strategy_label = DEFENSE_CONFIGS[strategy_enum].name
            except (ValueError, KeyError):
                strategy_label = strategy_key.replace('_', ' ').title()
            strategy_labels.append(strategy_label)

        # Calculate ASR for each strategy-model combination
        data = []
        for strategy in strategies:
            for model in models:
                stats = defense_model_stats[strategy][model]
                asr = (stats['success'] / stats['total']
                       * 100) if stats['total'] > 0 else 0
                data.append({'strategy': strategy, 'model': model, 'asr': asr})

        df = pd.DataFrame(data)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))

        # Set up bar positions
        x = np.arange(len(strategies))
        width = 0.8 / len(models)  # Width of each bar

        # Color palette for models
        model_colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316']

        # Plot bars for each model
        for i, model in enumerate(models):
            model_data = [df[(df['strategy'] == s) & (df['model'] == model)]['asr'].values[0]
                          if len(df[(df['strategy'] == s) & (df['model'] == model)]) > 0 else 0
                          for s in strategies]

            offset = (i - len(models)/2) * width + width/2
            bars = ax.bar(x + offset, model_data, width,
                          label=model, color=model_colors[i % len(
                              model_colors)],
                          alpha=0.8, edgecolor='black', linewidth=1)

            # Add value labels on bars
            for j, (bar, asr) in enumerate(zip(bars, model_data)):
                if asr > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{asr:.1f}%',
                            ha='center', va='bottom', fontsize=8, rotation=0)

        # Customize
        ax.set_ylabel('Attack Success Rate (%)',
                      fontsize=12, fontweight='bold')
        ax.set_xlabel('Defense Strategy', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_labels, rotation=45,
                           ha='right', fontsize=10)
        ax.set_ylim(0, max([d['asr'] for d in data]) * 1.2 if data else 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(title='Models', loc='upper right', fontsize=9)

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_all_charts(self) -> Dict[str, str]:
        """
        Create all available charts and return as base64 strings.

        Returns:
            Dictionary mapping chart name to base64-encoded PNG
        """
        return {
            'asr_bar_chart': self.create_asr_bar_chart(),
            'heatmap': self.create_heatmap(),
            'trend_line_chart': self.create_trend_line_chart(),
            'radar_chart': self.create_radar_chart(),
            'defense_comparison': self.create_defense_comparison_chart()
        }

    def create_summary_dashboard(
        self,
        title: str = "Prompt Injection Security Research Dashboard",
        as_base64: bool = True
    ) -> Any:
        """
        Create a multi-panel dashboard with all key visualizations.

        Args:
            title: Dashboard title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        fig = plt.figure(figsize=(20, 16))

        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Add title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # Get data
        model_metrics = self.collector.get_model_metrics()

        if not model_metrics:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'No data available\nRun some tests first!',
                    ha='center', va='center', fontsize=16)
            ax.axis('off')
            return self._fig_to_base64(fig) if as_base64 else fig

        # Panel 1: ASR Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        models = list(model_metrics.values())
        names = [m.model_name for m in models]
        asrs = [m.asr * 100 for m in models]
        colors = [self.colors['success'] if asr < 20 else
                  self.colors['warning'] if asr < 40 else
                  self.colors['danger'] for asr in asrs]
        bars = ax1.bar(range(len(names)), asrs, color=colors)
        ax1.set_ylabel('ASR (%)')
        ax1.set_title('Attack Success Rate by Model')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')

        # Panel 2: Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        model_names, categories, matrix = self.collector.get_heatmap_data()
        if matrix:
            data = np.array(matrix) * 100
            short_cats = [c.replace('_', ' ')[:15] for c in categories]
            short_models = [m[:15] for m in model_names]
            sns.heatmap(data, annot=True, fmt='.0f', cmap='RdYlGn_r',
                        xticklabels=short_cats, yticklabels=short_models, ax=ax2)
            ax2.set_title('Model × Attack Heatmap')
        else:
            ax2.text(0.5, 0.5, 'No heatmap data', ha='center', va='center')

        # Panel 3: Trend Line
        ax3 = fig.add_subplot(gs[1, 0])
        trends = self.collector.get_trend_data()
        if trends:
            for idx, (cat, values) in enumerate(sorted(trends.items())):
                x = range(1, len(values) + 1)
                y = [v * 100 for v in values]
                ax3.plot(x, y, marker='o', label=cat.replace('_', ' ')[:15])
            ax3.legend(loc='best', fontsize=8)
            ax3.set_xlabel('Variations Tested')
            ax3.set_ylabel('Cumulative ASR (%)')
            ax3.set_title('ASR Trend by Attack Category')
        else:
            ax3.text(0.5, 0.5, 'No trend data', ha='center', va='center')

        # Panel 4: Summary Stats
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        summary = self.collector.get_summary()
        summary_text = f"""
SUMMARY STATISTICS

Total Tests: {summary.get('total_tests', 0)}
Models Tested: {summary.get('unique_models', 0)}
Attack Categories: {summary.get('unique_attack_categories', 0)}

Overall ASR: {summary.get('overall_asr', 0)*100:.1f}%

Most Robust Model: {summary.get('most_robust_model', 'N/A')}
Most Vulnerable Model: {summary.get('most_vulnerable_model', 'N/A')}

Most Effective Attack: {summary.get('most_effective_attack', 'N/A')}
Least Effective Attack: {summary.get('least_effective_attack', 'N/A')}
"""
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        ax4.set_title('Summary')

        # Suppress tight_layout warnings for incompatible axes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    # =========================================================================
    # VISUALIZATIONS (2×4 Matrix Framework)
    # =========================================================================
    # These visualizations support the lightweight framework that decomposes
    # attacks by injection vector (direct/indirect) and attack objective.

    def create_taxonomy_heatmap(
        self,
        title: str = "Attack Success Rate: Matrix (2×3)",
        as_base64: bool = True
    ) -> Any:
        """
        Create a heatmap showing ASR for the 2×4 matrix.

        Rows: Injection Vector (Direct, Indirect)
        Columns: Attack Objective (Instruction Override, Data Extraction, 
                                   Role Confusion, Tool Misuse)

        This visualization helps identify:
        - Which vector × objective combinations are most vulnerable
        - Differences between direct and indirect injection effectiveness
        - Which attack objectives are hardest to defend against

        Args:
            title: Chart title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        vectors, objectives, matrix = self.collector.get_taxonomy_heatmap_data()

        if not matrix or not any(any(row) for row in matrix):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No matrix data available\nRun tests with matrix-based attacks',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Convert to percentage
        data = np.array(matrix) * 100

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Custom colormap: green (safe) to red (vulnerable)
        cmap = sns.diverging_palette(145, 10, s=80, l=55, as_cmap=True)

        # Create heatmap
        im = sns.heatmap(
            data,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            xticklabels=objectives,
            yticklabels=vectors,
            cbar_kws={'label': 'Attack Success Rate (%)'},
            linewidths=2,
            linecolor='white',
            ax=ax,
            vmin=0,
            vmax=100,
            annot_kws={'size': 14, 'weight': 'bold'}
        )

        # Styling
        ax.set_xlabel('Attack Objective', fontsize=14, fontweight='bold')
        ax.set_ylabel('Injection Vector', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Make y-axis labels horizontal
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                           ha='right', fontsize=11)

        # Add annotation about success definitions
        success_text = (
            "Success Definitions:\n"
            "• Instruction Override: Policy violated\n"
            "• Data Extraction: Secret leaked\n"
            "• Role Confusion: Wrong role assumed"
        )
        fig.text(0.02, 0.02, success_text, fontsize=9,
                 verticalalignment='bottom', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_vector_comparison_chart(
        self,
        title: str = "Direct vs Indirect Injection Comparison",
        as_base64: bool = True
    ) -> Any:
        """
        Create a grouped bar chart comparing direct vs indirect injection ASR.

        Shows side-by-side comparison for each attack objective.

        Args:
            title: Chart title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        comparison = self.collector.get_vector_comparison()

        if not comparison or all(c['total_tests'] == 0 for c in comparison.values()):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No comparison data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Prepare data
        objectives = ['instruction_override',
                      'data_extraction', 'role_confusion']
        objective_labels = ['Instruction\nOverride',
                            'Data\nExtraction', 'Role\nConfusion']

        direct_asrs = [comparison['direct']['objectives'].get(
            o, 0) * 100 for o in objectives]
        indirect_asrs = [comparison['indirect']
                         ['objectives'].get(o, 0) * 100 for o in objectives]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(objectives))
        width = 0.35

        # Create grouped bars
        bars1 = ax.bar(x - width/2, direct_asrs, width, label='Direct Injection',
                       color=self.colors['primary'], edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, indirect_asrs, width, label='Indirect Injection',
                       color=self.colors['warning'], edgecolor='white', linewidth=1.5)

        # Add value labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        add_labels(bars1)
        add_labels(bars2)

        # Styling
        ax.set_xlabel('Attack Objective', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)',
                      fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(objective_labels, fontsize=11)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_ylim(0, max(max(direct_asrs), max(indirect_asrs)) *
                    1.2 if any(direct_asrs + indirect_asrs) else 100)
        ax.grid(True, alpha=0.3, axis='y')

        # Add overall ASR annotation
        direct_overall = comparison['direct']['overall_asr'] * 100
        indirect_overall = comparison['indirect']['overall_asr'] * 100
        overall_text = f"Overall ASR:\nDirect: {direct_overall:.1f}%\nIndirect: {indirect_overall:.1f}%"
        ax.text(0.98, 0.98, overall_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_defense_effectiveness_chart(
        self,
        baseline_results,
        defense_name: str = "Defense",
        title: str = None,
        as_base64: bool = True
    ) -> Any:
        """
        Create a chart showing Defense Effectiveness (DE) across matrix cells.

        DE = 1 - (ASR_defended / ASR_baseline)

        Args:
            baseline_results: Results from baseline (no defense) testing
            defense_name: Name of the defense being evaluated
            title: Chart title (auto-generated if None)
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        if title is None:
            title = f"Defense Effectiveness: {defense_name}"

        de_matrix = self.collector.calculate_defense_effectiveness(
            baseline_results, defense_name)

        if not de_matrix:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Cannot calculate DE without baseline data',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return self._fig_to_base64(fig) if as_base64 else fig

        # Convert to 2D array for heatmap
        vectors = ['direct', 'indirect']
        objectives = ['instruction_override',
                      'data_extraction', 'role_confusion']

        data = []
        for v in vectors:
            row = []
            for o in objectives:
                de = de_matrix[v][o]
                row.append(de * 100 if de is not None else np.nan)
            data.append(row)

        data = np.array(data)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Custom colormap: red (negative DE) to green (positive DE)
        cmap = sns.diverging_palette(10, 145, s=80, l=55, as_cmap=True)

        # Create heatmap
        sns.heatmap(
            data,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            xticklabels=['Instruction\nOverride',
                         'Data\nExtraction', 'Role\nConfusion'],
            yticklabels=['Direct', 'Indirect'],
            cbar_kws={'label': 'Defense Effectiveness (%)'},
            linewidths=2,
            linecolor='white',
            ax=ax,
            center=0,
            annot_kws={'size': 12, 'weight': 'bold'},
            mask=np.isnan(data)
        )

        # Styling
        ax.set_xlabel('Attack Objective', fontsize=14, fontweight='bold')
        ax.set_ylabel('Injection Vector', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

        # Add interpretation guide
        guide_text = (
            "Interpretation:\n"
            "• DE > 0: Defense reduces ASR\n"
            "• DE = 100%: Attack fully blocked\n"
            "• DE < 0: Defense increases vulnerability"
        )
        fig.text(0.02, 0.02, guide_text, fontsize=9,
                 verticalalignment='bottom', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

        plt.tight_layout()

        return self._fig_to_base64(fig) if as_base64 else fig

    def create_taxonomy_dashboard(
        self,
        title: str = "Analysis Dashboard",
        as_base64: bool = True
    ) -> Any:
        """
        Create a comprehensive dashboard for matrix-based analysis.

        Includes:
        - 2×4 matrix heatmap
        - Vector comparison chart
        - Objective comparison chart
        - Summary statistics

        Args:
            title: Dashboard title
            as_base64: If True, return base64 string; else return Figure

        Returns:
            Base64 string or matplotlib Figure
        """
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

        # Panel 1: Matrix Heatmap (2×4)
        ax1 = fig.add_subplot(gs[0, :])
        vectors, objectives, matrix = self.collector.get_taxonomy_heatmap_data()

        if matrix and any(any(row) for row in matrix):
            data = np.array(matrix) * 100
            cmap = sns.diverging_palette(145, 10, s=80, l=55, as_cmap=True)
            sns.heatmap(data, annot=True, fmt='.1f', cmap=cmap,
                        xticklabels=objectives, yticklabels=vectors,
                        cbar_kws={'label': 'ASR (%)'}, linewidths=1.5,
                        ax=ax1, vmin=0, vmax=100, annot_kws={'size': 12, 'weight': 'bold'})
            ax1.set_title(
                'Matrix: Injection Vector × Attack Objective', fontsize=14, pad=10)
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
        else:
            ax1.text(0.5, 0.5, 'No matrix data available',
                     ha='center', va='center')
            ax1.set_title('Matrix')

        # Panel 2: Vector Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        comparison = self.collector.get_vector_comparison()

        if comparison:
            labels = ['Direct', 'Indirect']
            asrs = [comparison['direct']['overall_asr'] * 100,
                    comparison['indirect']['overall_asr'] * 100]
            colors = [self.colors['primary'], self.colors['warning']]
            bars = ax2.bar(labels, asrs, color=colors,
                           edgecolor='white', linewidth=1.5)
            for bar, asr in zip(bars, asrs):
                ax2.annotate(f'{asr:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')
            ax2.set_ylabel('ASR (%)')
            ax2.set_title('Overall ASR by Injection Vector', fontsize=12)
            ax2.set_ylim(0, max(asrs) * 1.3 if any(asrs) else 100)
        else:
            ax2.text(0.5, 0.5, 'No vector data', ha='center', va='center')

        # Panel 3: Objective Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        obj_comparison = self.collector.get_objective_comparison()

        if obj_comparison:
            obj_labels = [o.replace('_', ' ').title()[:12]
                          for o in obj_comparison.keys()]
            obj_asrs = [obj_comparison[o]['overall_asr']
                        * 100 for o in obj_comparison.keys()]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(obj_labels)))
            bars = ax3.bar(obj_labels, obj_asrs, color=colors,
                           edgecolor='white', linewidth=1.5)
            for bar, asr in zip(bars, obj_asrs):
                ax3.annotate(f'{asr:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
            ax3.set_ylabel('ASR (%)')
            ax3.set_title('Overall ASR by Attack Objective', fontsize=12)
            ax3.tick_params(axis='x', rotation=30)
            ax3.set_ylim(0, max(obj_asrs) * 1.3 if any(obj_asrs) else 100)
        else:
            ax3.text(0.5, 0.5, 'No objective data', ha='center', va='center')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        return self._fig_to_base64(fig) if as_base64 else fig


_visualizer: Optional[Visualizer] = None


def get_visualizer() -> Visualizer:
    """Get the global Visualizer instance."""
    global _visualizer
    if _visualizer is None:
        _visualizer = Visualizer()
    return _visualizer
