from PIL import Image
import seaborn as sns
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import io

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Defense strategy display names mapping
DEFENSE_DISPLAY_NAMES = {
    'none': 'Baseline',
    'strong_prefix': 'System Prompt',
    'source_tagging': 'Source Tag',
    'output_filtering': 'Output Filter',
    'combined_all': 'Combined'
}

# Objective short names mapping
OBJECTIVE_SHORT_NAMES = {
    'Instruction Override': 'Instr. Override',
    'Data Extraction': 'Data Extraction',
    'Role Confusion': 'Role Confusion'
}


def get_results_files() -> List[str]:
    """Get list of all results CSV files in the results directory."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []

    csv_files = list(results_dir.glob("results_*.csv"))
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(f) for f in csv_files]


def get_file_choices() -> List[str]:
    """Get file choices for dropdown."""
    files = get_results_files()
    return [Path(f).name for f in files] if files else ["No results files found"]


def load_results_data(filename: str) -> Optional[pd.DataFrame]:
    """Load results from a CSV file."""
    if not filename or filename == "No results files found":
        return None

    filepath = Path("results") / filename
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} results from {filename}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return None


def calculate_asr(df: pd.DataFrame) -> float:
    """Calculate Attack Success Rate from dataframe."""
    if df is None or df.empty:
        return 0.0

    success_count = df[df['detection_result'] == 'success'].shape[0]
    total_count = df.shape[0]

    return success_count / total_count if total_count > 0 else 0.0


def parse_attack_category(category: str) -> Tuple[str, str]:
    """Parse attack category into vector and objective."""
    category = str(category)

    # Determine injection vector
    if category.startswith('Indirect'):
        vector = 'Indirect'
    elif category.startswith('Direct'):
        vector = 'Direct'
    else:
        vector = 'Unknown'

    # Extract objective (remove Direct/Indirect prefix)
    objective = category.replace('Direct ', '').replace('Indirect ', '')

    return vector, objective


def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image for Gradio."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def get_defense_display_name(strategy: str) -> str:
    """Get display name for a defense strategy."""
    return DEFENSE_DISPLAY_NAMES.get(strategy, strategy.replace('_', ' ').title())


def create_baseline_asr_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Create Baseline ASR Table (Vector × Objective × Model).
    Shows ASR breakdown for baseline (no defense) data only.
    """
    if df is None or df.empty:
        return None

    # Filter to baseline (none) defense only
    baseline_df = df[df['defense_strategy'] == 'none']

    if baseline_df.empty:
        logger.warning("No baseline data found")
        return None

    models = baseline_df['model_name'].unique().tolist()
    short_models = [m.split(' (')[0].strip() for m in models]

    # Build table data
    table_data = []

    vectors = ['Direct', 'Indirect']
    objectives = ['Instruction Override', 'Data Extraction', 'Role Confusion']

    for vector in vectors:
        for obj_idx, objective in enumerate(objectives):
            row = {
                'Vector': vector if obj_idx == 0 else '',
                'Objective': OBJECTIVE_SHORT_NAMES.get(objective, objective)
            }

            # Calculate ASR for each model
            total_success = 0
            total_tests = 0

            for model, short_model in zip(models, short_models):
                subset = baseline_df[
                    (baseline_df['model_name'] == model) &
                    (baseline_df['attack_category'] == f'{vector} {objective}')
                ]
                asr = calculate_asr(subset) * 100
                row[short_model] = f'{asr:.0f}%'

                total_success += (subset['detection_result']
                                  == 'success').sum()
                total_tests += len(subset)

            # Calculate total ASR
            total_asr = (total_success / total_tests *
                         100) if total_tests > 0 else 0
            row['Total'] = f'{total_asr:.0f}%'

            table_data.append(row)

    return pd.DataFrame(table_data)


def create_defense_effectiveness_chart(df: pd.DataFrame, vector: str = 'All'):
    """
    Create Defense Effectiveness Grouped Bar Chart.
    X-axis: Three attack objectives
    Y-axis: ASR (0-100%)
    Bars: Baseline, System Prompt, Source Tag, Output Filter, Combined

    Args:
        df: DataFrame with results
        vector: 'All', 'Direct', or 'Indirect' - which injection vector to show
    """
    if df is None or df.empty:
        return None

    strategies = ['none', 'strong_prefix', 'source_tagging',
                  'output_filtering', 'combined_all']
    objectives = ['Instruction Override', 'Data Extraction', 'Role Confusion']

    # Check which strategies exist in data
    available_strategies = [
        s for s in strategies if s in df['defense_strategy'].unique()]

    if len(available_strategies) < 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Defense Effectiveness requires multiple defense strategies.\nRun batch testing with defenses first.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig_to_image(fig)

    # Build ASR data for each objective and strategy
    asr_data = {obj: [] for obj in objectives}

    for objective in objectives:
        for strategy in available_strategies:
            # Filter by vector type
            if vector == 'Direct':
                subset = df[
                    (df['defense_strategy'] == strategy) &
                    (df['attack_category'] == f'Direct {objective}')
                ]
            elif vector == 'Indirect':
                subset = df[
                    (df['defense_strategy'] == strategy) &
                    (df['attack_category'] == f'Indirect {objective}')
                ]
            else:  # 'All'
                subset = df[
                    (df['defense_strategy'] == strategy) &
                    (df['attack_category'].str.contains(objective))
                ]
            asr = calculate_asr(subset) * 100
            asr_data[objective].append(asr)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(objectives))
    width = 0.15
    n_strategies = len(available_strategies)

    # Color palette
    colors = ['#95a5a6', '#3498db', '#2ecc71',
              '#e74c3c', '#9b59b6'][:n_strategies]

    for i, strategy in enumerate(available_strategies):
        offset = (i - n_strategies/2 + 0.5) * width
        values = [asr_data[obj][i] for obj in objectives]
        display_name = get_defense_display_name(strategy)
        bars = ax.bar(x + offset, values, width,
                      label=display_name, color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Attack Objective', fontsize=12, fontweight='bold')
    ax.set_ylabel('ASR (%)', fontsize=12, fontweight='bold')

    # Update title based on vector
    if vector == 'All':
        title = 'Defense Effectiveness by Attack Objective (All Vectors)'
    else:
        title = f'Defense Effectiveness by Attack Objective ({vector} Injection)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(objectives, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig_to_image(fig)


def create_defense_impact_matrix_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Create Defense Impact Matrix Table.
    Shows baseline ASR and reduction for each defense strategy.
    """
    if df is None or df.empty:
        return None

    strategies = ['none', 'strong_prefix', 'source_tagging',
                  'output_filtering', 'combined_all']
    available_strategies = [
        s for s in strategies if s in df['defense_strategy'].unique()]

    if 'none' not in available_strategies:
        logger.warning("No baseline data found for impact matrix")
        return None

    defense_strategies = [s for s in available_strategies if s != 'none']

    if not defense_strategies:
        return None

    vectors = ['Direct', 'Indirect']
    objectives = ['Instruction Override', 'Data Extraction', 'Role Confusion']

    table_data = []

    for vector in vectors:
        for obj_idx, objective in enumerate(objectives):
            category = f'{vector} {objective}'

            row = {
                'Vector': vector if obj_idx == 0 else '',
                'Objective': OBJECTIVE_SHORT_NAMES.get(objective, objective)
            }

            # Get baseline ASR
            baseline_subset = df[
                (df['defense_strategy'] == 'none') &
                (df['attack_category'] == category)
            ]
            baseline_asr = calculate_asr(baseline_subset) * 100
            row['Baseline'] = f'{baseline_asr:.0f}%'

            # Get ASR for each defense strategy
            for strategy in defense_strategies:
                defense_subset = df[
                    (df['defense_strategy'] == strategy) &
                    (df['attack_category'] == category)
                ]
                defense_asr = calculate_asr(defense_subset) * 100

                # Calculate reduction percentage
                if baseline_asr > 0:
                    reduction = ((baseline_asr - defense_asr) /
                                 baseline_asr) * 100
                else:
                    reduction = 0

                display_name = f'+{get_defense_display_name(strategy)}'
                row[display_name] = f'{defense_asr:.0f}% ({reduction:+.0f}%)'

            table_data.append(row)

    return pd.DataFrame(table_data)


def create_performance_overhead_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Create Performance Overhead by Defense Table.
    Shows latency, tokens, and ASR metrics for each defense.
    """
    if df is None or df.empty:
        return None

    strategies = ['none', 'strong_prefix', 'source_tagging',
                  'output_filtering', 'combined_all']
    available_strategies = [
        s for s in strategies if s in df['defense_strategy'].unique()]

    if 'none' not in available_strategies or len(available_strategies) < 2:
        return None

    # Get baseline metrics
    baseline_df = df[df['defense_strategy'] == 'none']
    baseline_latency = baseline_df['response_time'].mean()
    baseline_tokens = baseline_df['tokens_generated'].mean()
    baseline_asr = calculate_asr(baseline_df) * 100

    table_data = []

    for strategy in available_strategies:
        strategy_df = df[df['defense_strategy'] == strategy]

        mean_latency = strategy_df['response_time'].mean()
        mean_tokens = strategy_df['tokens_generated'].mean()
        mean_asr = calculate_asr(strategy_df) * 100

        display_name = get_defense_display_name(strategy)
        if strategy != 'none':
            display_name = f'+{display_name}'

        row = {
            'Defense': display_name,
            'Mean Latency (s)': f'{mean_latency:.1f}s',
        }

        # Calculate increases relative to baseline
        if strategy == 'none':
            row['Latency ↑'] = '-'
            row['Mean Tokens'] = f'{mean_tokens:.0f}'
            row['Tokens ↑'] = '-'
            row['Mean ASR ↓'] = f'{mean_asr:.0f}%'
        else:
            latency_increase = ((mean_latency - baseline_latency) /
                                baseline_latency) * 100 if baseline_latency > 0 else 0
            tokens_increase = ((mean_tokens - baseline_tokens) /
                               baseline_tokens) * 100 if baseline_tokens > 0 else 0
            asr_reduction = baseline_asr - mean_asr

            row['Latency ↑'] = f'+{latency_increase:.0f}%' if latency_increase >= 0 else f'{latency_increase:.0f}%'
            row['Mean Tokens'] = f'{mean_tokens:.0f}'
            row['Tokens ↑'] = f'+{tokens_increase:.0f}%' if tokens_increase >= 0 else f'{tokens_increase:.0f}%'
            row['Mean ASR ↓'] = f'-{asr_reduction:.0f}%' if asr_reduction >= 0 else f'+{abs(asr_reduction):.0f}%'

        table_data.append(row)

    return pd.DataFrame(table_data)


def create_pareto_front_chart(df: pd.DataFrame):
    """
    Create Security-Performance Pareto Front scatter plot.
    X-axis: Mean Latency Increase (%)
    Y-axis: Mean ASR Reduction (%)
    Bubble size: Token Increase (%)
    """
    if df is None or df.empty:
        return None

    strategies = ['strong_prefix', 'source_tagging',
                  'output_filtering', 'combined_all']
    available_strategies = [
        s for s in strategies if s in df['defense_strategy'].unique()]

    if not available_strategies or 'none' not in df['defense_strategy'].unique():
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'Pareto Front requires baseline and defense data.\nRun batch testing with defenses first.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig_to_image(fig)

    # Get baseline metrics
    baseline_df = df[df['defense_strategy'] == 'none']
    baseline_latency = baseline_df['response_time'].mean()
    baseline_tokens = baseline_df['tokens_generated'].mean()
    baseline_asr = calculate_asr(baseline_df) * 100

    # Calculate metrics for each defense
    plot_data = []

    for strategy in available_strategies:
        strategy_df = df[df['defense_strategy'] == strategy]

        mean_latency = strategy_df['response_time'].mean()
        mean_tokens = strategy_df['tokens_generated'].mean()
        mean_asr = calculate_asr(strategy_df) * 100

        latency_increase = ((mean_latency - baseline_latency) /
                            baseline_latency) * 100 if baseline_latency > 0 else 0
        tokens_increase = ((mean_tokens - baseline_tokens) /
                           baseline_tokens) * 100 if baseline_tokens > 0 else 0
        asr_reduction = baseline_asr - mean_asr

        plot_data.append({
            'strategy': strategy,
            'display_name': get_defense_display_name(strategy),
            'latency_increase': latency_increase,
            'asr_reduction': asr_reduction,
            # Minimum size for visibility
            'tokens_increase': max(tokens_increase, 5)
        })

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {'strong_prefix': '#3498db', 'source_tagging': '#2ecc71',
              'output_filtering': '#e74c3c', 'combined_all': '#9b59b6'}

    for data in plot_data:
        size = max(data['tokens_increase'] * 10, 100)  # Scale bubble size
        ax.scatter(
            data['latency_increase'],
            data['asr_reduction'],
            s=size,
            c=colors.get(data['strategy'], '#95a5a6'),
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            label=f"{data['display_name']} (Tokens ↑: {data['tokens_increase']:.0f}%)"
        )

        # Add label
        ax.annotate(
            data['display_name'],
            (data['latency_increase'], data['asr_reduction']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Ideal region annotation (top-left: low latency increase, high ASR reduction)
    ax.annotate('Better Security →', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, color='green', ha='left', va='top')
    ax.annotate('← Lower Cost', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, color='blue', ha='left', va='bottom')

    ax.set_xlabel('Mean Latency Increase (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean ASR Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Security-Performance Pareto Front\n(Bubble size = Token Increase %)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_image(fig)


def generate_summary(df: pd.DataFrame, filename: str) -> str:
    """Generate markdown summary of the results."""
    if df is None or df.empty:
        return "No data available. Select a results file."

    total_tests = len(df)
    unique_models = df['model_name'].nunique()
    unique_attacks = df['attack_category'].nunique()
    unique_defenses = df['defense_strategy'].nunique()
    overall_asr = calculate_asr(df) * 100

    # Get baseline ASR
    baseline_df = df[df['defense_strategy'] == 'none']
    baseline_asr = calculate_asr(baseline_df) * \
        100 if not baseline_df.empty else 0

    # Get best defense ASR
    best_defense = None
    best_asr = 100
    for strategy in df['defense_strategy'].unique():
        if strategy != 'none':
            strategy_df = df[df['defense_strategy'] == strategy]
            strategy_asr = calculate_asr(strategy_df) * 100
            if strategy_asr < best_asr:
                best_asr = strategy_asr
                best_defense = strategy

    summary = f"""
### Results Overview

| Metric | Value |
|--------|-------|
| **Total Tests** | {total_tests} |
| **Models Tested** | {unique_models} |
| **Attack Categories** | {unique_attacks} |
| **Defense Strategies** | {unique_defenses} |
| **Baseline ASR** | {baseline_asr:.1f}% |
"""

    if best_defense:
        best_defense_name = get_defense_display_name(best_defense)
        improvement = baseline_asr - best_asr
        summary += f"| **Best Defense** | {best_defense_name} ({best_asr:.1f}%, -{improvement:.1f}%) |\n"

    return summary


def refresh_visualizations(selected_file: str):
    """Refresh all visualizations based on selected file."""
    df = load_results_data(selected_file)

    if df is None:
        empty_msg = "No data available. Select a valid results file."
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        empty_img = fig_to_image(fig)
        return empty_msg, None, empty_img, empty_img, empty_img, None, None, empty_img

    summary = generate_summary(df, selected_file)
    baseline_table = create_baseline_asr_table(df)
    defense_chart_all = create_defense_effectiveness_chart(df, vector='All')
    defense_chart_direct = create_defense_effectiveness_chart(
        df, vector='Direct')
    defense_chart_indirect = create_defense_effectiveness_chart(
        df, vector='Indirect')
    impact_table = create_defense_impact_matrix_table(df)
    performance_table = create_performance_overhead_table(df)
    pareto_chart = create_pareto_front_chart(df)

    return summary, baseline_table, defense_chart_all, defense_chart_direct, defense_chart_indirect, impact_table, performance_table, pareto_chart


def create_visualization_tab():
    """Create the Visualization tab UI component."""

    with gr.TabItem("Visualization", id="viz"):
        gr.Markdown("### Batch Testing Results Visualization")

        with gr.Row():
            file_dropdown = gr.Dropdown(
                choices=get_file_choices(),
                value=get_file_choices()[0] if get_file_choices() and get_file_choices()[
                    0] != "No results files found" else None,
                label="Select Results File",
                interactive=True,
                scale=2
            )
            reload_btn = gr.Button("🔄 Reload Files")
            load_btn = gr.Button("📊 Load Data", variant="primary")

        # Summary section
        summary_md = gr.Markdown(
            "Select a results file to view visualizations.")

        # 1. Baseline ASR Table
        gr.Markdown("---")
        gr.Markdown("### 1. Baseline ASR Table (Vector × Objective × Model)")
        gr.Markdown("*Shows attack success rates with no defense applied*")
        baseline_table = gr.Dataframe(
            label="",
            interactive=False,
            wrap=True
        )

        # 2. Defense Effectiveness Chart
        gr.Markdown("---")
        gr.Markdown("### 2. Defense Effectiveness by Attack Objective")
        gr.Markdown(
            "*Grouped bar chart comparing ASR across defense strategies*")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### All Vectors Combined")
                defense_chart_all = gr.Image(label="")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Direct Injection Only")
                defense_chart_direct = gr.Image(label="")
            with gr.Column():
                gr.Markdown("#### Indirect Injection Only")
                defense_chart_indirect = gr.Image(label="")

        # 3. Defense Impact Matrix Table
        gr.Markdown("---")
        gr.Markdown(
            "### 3. Defense Impact Matrix (Baseline → Defense Strategies)")
        gr.Markdown(
            "*Shows ASR reduction for each defense relative to baseline*")
        impact_table = gr.Dataframe(
            label="",
            interactive=False,
            wrap=True
        )

        # 4. Performance Overhead Table
        gr.Markdown("---")
        gr.Markdown("### 4. Performance Overhead by Defense")
        gr.Markdown("*Compares latency, token usage, and security improvement*")
        performance_table = gr.Dataframe(
            label="",
            interactive=False,
            wrap=True
        )

        # 5. Security-Performance Pareto Front
        gr.Markdown("---")
        gr.Markdown("### 5. Security-Performance Pareto Front")
        gr.Markdown("*Trade-off visualization: latency cost vs security gain*")
        pareto_chart = gr.Image(label="")

        # Event handlers
        file_dropdown.change(
            fn=refresh_visualizations,
            inputs=[file_dropdown],
            outputs=[summary_md, baseline_table, defense_chart_all, defense_chart_direct,
                     defense_chart_indirect, impact_table, performance_table, pareto_chart]
        )

        reload_btn.click(
            fn=lambda: gr.Dropdown(
                choices=get_file_choices(),
                value=get_file_choices()[0] if get_file_choices() and get_file_choices()[
                    0] != "No results files found" else None
            ),
            inputs=None,
            outputs=[file_dropdown]
        )

        load_btn.click(
            fn=refresh_visualizations,
            inputs=[file_dropdown],
            outputs=[summary_md, baseline_table, defense_chart_all, defense_chart_direct,
                     defense_chart_indirect, impact_table, performance_table, pareto_chart]
        )
