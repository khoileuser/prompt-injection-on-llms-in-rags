from PIL import Image
import seaborn as sns
import numpy as np
import logging
from typing import Optional, List
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


def get_results_files() -> List[str]:
    """Get list of all results CSV files in the results directory."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []

    csv_files = list(results_dir.glob("*_results_*.csv"))
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(f) for f in csv_files]


def get_attack_file_choices() -> List[str]:
    """Get attack results file choices for dropdown."""
    files = get_results_files()
    attack_files = [Path(f).name for f in files if "attack_results" in f]
    return attack_files if attack_files else ["No attack results files found"]


def get_defense_file_choices() -> List[str]:
    """Get defense results file choices for dropdown."""
    files = get_results_files()
    defense_files = [Path(f).name for f in files if "defense_results" in f]
    return defense_files if defense_files else ["No defense results files found"]


def get_file_choices() -> List[str]:
    """Get file choices for dropdown (legacy, returns all files)."""
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


def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image for Gradio."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def generate_summary(df: pd.DataFrame, filename: str) -> str:
    """Generate markdown summary of the results."""
    if df is None or df.empty:
        return "No data available. Select a results file."

    test_type = "Attack Testing" if "attack_results" in filename else "Defense Testing"

    total_tests = len(df)
    unique_models = df['model_name'].nunique()
    unique_attacks = df['attack_category'].nunique()
    unique_defenses = df['defense_strategy'].nunique()
    overall_asr = calculate_asr(df) * 100

    model_asr = {}
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        model_asr[model] = calculate_asr(model_df) * 100

    sorted_models = sorted(model_asr.items(), key=lambda x: x[1])
    most_robust = sorted_models[0][0] if sorted_models else "N/A"
    most_vulnerable = sorted_models[-1][0] if sorted_models else "N/A"

    summary = f"""
## Test Statistics
**Data Source:** {filename} ({test_type})

| Metric | Value |
|--------|-------|
| Total Tests | {total_tests} |
| Models Tested | {unique_models} |
| Attack Categories | {unique_attacks} |
| Defense Strategies | {unique_defenses} |
| Overall ASR | {overall_asr:.1f}% |

## Key Findings
- **Most Robust Model:** {most_robust} ({model_asr.get(most_robust, 0):.1f}% ASR)
- **Most Vulnerable Model:** {most_vulnerable} ({model_asr.get(most_vulnerable, 0):.1f}% ASR)
"""
    return summary


def create_model_asr_chart(df: pd.DataFrame):
    """Create bar chart for ASR by model."""
    if df is None or df.empty:
        return None

    model_data = []
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        asr = calculate_asr(model_df) * 100
        short_name = model.split('(')[0].strip()
        model_data.append({'Model': short_name, 'ASR': asr})

    model_df = pd.DataFrame(model_data).sort_values('ASR')

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if asr < 30 else '#f39c12' if asr <
              60 else '#e74c3c' for asr in model_df['ASR']]
    bars = ax.barh(model_df['Model'], model_df['ASR'], color=colors)

    ax.set_xlabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('ASR by Model', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)

    for bar, asr in zip(bars, model_df['ASR']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{asr:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    return fig_to_image(fig)


def create_attack_category_chart(df: pd.DataFrame):
    """Create bar chart for ASR by attack category."""
    if df is None or df.empty:
        return None

    attack_data = []
    for attack in df['attack_category'].unique():
        attack_df = df[df['attack_category'] == attack]
        asr = calculate_asr(attack_df) * 100
        attack_data.append({'Attack': attack, 'ASR': asr})

    attack_df = pd.DataFrame(attack_data).sort_values('ASR', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if asr > 60 else '#f39c12' if asr >
              30 else '#2ecc71' for asr in attack_df['ASR']]
    bars = ax.bar(range(len(attack_df)), attack_df['ASR'], color=colors)

    ax.set_xticks(range(len(attack_df)))
    ax.set_xticklabels(attack_df['Attack'], rotation=45, ha='right')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('ASR by Attack Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    for bar, asr in zip(bars, attack_df['ASR']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{asr:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    return fig_to_image(fig)


def create_heatmap(df: pd.DataFrame):
    """Create heatmap of Model x Attack Category ASR."""
    if df is None or df.empty:
        return None

    models = df['model_name'].unique().tolist()
    attacks = df['attack_category'].unique().tolist()

    matrix = []
    for model in models:
        row = []
        for attack in attacks:
            subset = df[(df['model_name'] == model) &
                        (df['attack_category'] == attack)]
            asr = calculate_asr(subset) * 100
            row.append(asr)
        matrix.append(row)

    short_models = [m.split('(')[0].strip() for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.set_yticks(range(len(short_models)))
    ax.set_yticklabels(short_models)

    for i in range(len(models)):
        for j in range(len(attacks)):
            text = ax.text(j, i, f'{matrix[i][j]:.1f}%', ha='center', va='center',
                           color='white' if matrix[i][j] > 50 else 'black', fontsize=10)

    ax.set_title('Model x Attack Category Heatmap (ASR %)',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='ASR (%)')
    plt.tight_layout()
    return fig_to_image(fig)


def create_defense_effectiveness_chart(df: pd.DataFrame):
    """Create grouped bar chart for Defense Effectiveness."""
    if df is None or df.empty:
        return None

    strategies = df['defense_strategy'].unique().tolist()
    if len(strategies) <= 1 or 'none' not in strategies:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Defense Effectiveness requires multiple defense strategies.\nRun Defense Testing first.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Defense Effectiveness', fontsize=14, fontweight='bold')
        return fig_to_image(fig)

    models = df['model_name'].unique().tolist()
    defense_strategies = [s for s in strategies if s != 'none']
    short_models = [m.split('(')[0].strip() for m in models]

    de_matrix = []
    for strategy in defense_strategies:
        de_row = []
        for model in models:
            baseline_df = df[(df['model_name'] == model) &
                             (df['defense_strategy'] == 'none')]
            baseline_asr = calculate_asr(baseline_df)

            defended_df = df[(df['model_name'] == model) & (
                df['defense_strategy'] == strategy)]
            defended_asr = calculate_asr(defended_df)

            if baseline_asr > 0:
                de = (1 - (defended_asr / baseline_asr)) * 100
            else:
                de = 100 if defended_asr == 0 else 0
            de_row.append(de)
        de_matrix.append(de_row)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(defense_strategies))
    width = 0.8 / len(models)

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(short_models, colors)):
        values = [de_matrix[j][i] for j in range(len(defense_strategies))]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=color)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Defense Strategy', fontsize=12)
    ax.set_ylabel('Defense Effectiveness (%)', fontsize=12)
    ax.set_title('Defense Effectiveness by Strategy\nDE = 1 - (ASR_defended / ASR_baseline) | Higher is better',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in defense_strategies])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(min(min(row) for row in de_matrix) - 10,
                max(max(row) for row in de_matrix) + 10)

    plt.tight_layout()
    return fig_to_image(fig)


def create_defense_asr_comparison(df: pd.DataFrame):
    """Create line chart comparing ASR across defense strategies."""
    if df is None or df.empty:
        return None

    strategies = df['defense_strategy'].unique().tolist()
    if len(strategies) <= 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Comparison requires multiple defense strategies.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('ASR by Defense Strategy', fontsize=14, fontweight='bold')
        return fig_to_image(fig)

    models = df['model_name'].unique().tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, model in enumerate(models):
        asr_values = []
        for strategy in strategies:
            subset = df[(df['model_name'] == model) & (
                df['defense_strategy'] == strategy)]
            asr = calculate_asr(subset) * 100
            asr_values.append(asr)

        short_name = model.split('(')[0].strip()
        ax.plot(range(len(strategies)), asr_values, marker=markers[i % len(markers)],
                label=short_name, color=colors[i], linewidth=2, markersize=8)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.capitalize() for s in strategies])
    ax.set_xlabel('Defense Strategy', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('ASR Comparison Across Defense Strategies',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_image(fig)


def create_results_breakdown_pie(df: pd.DataFrame):
    """Create pie chart for detection results breakdown."""
    if df is None or df.empty:
        return None

    result_counts = df['detection_result'].value_counts()

    colors_map = {
        'blocked': '#2ecc71',
        'success': '#e74c3c',
        'error': '#95a5a6'
    }
    colors = [colors_map.get(r, '#95a5a6') for r in result_counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(result_counts.values, labels=result_counts.index,
                                      autopct='%1.1f%%', colors=colors,
                                      wedgeprops=dict(width=0.6), pctdistance=0.75)

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax.set_title('Overall Results Breakdown', fontsize=14, fontweight='bold')

    centre_circle = plt.Circle((0, 0), 0.40, fc='white')
    ax.add_patch(centre_circle)

    total = sum(result_counts.values)
    ax.text(0, 0, f'Total\n{total}', ha='center',
            va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig_to_image(fig)


def refresh_attack_visualizations(selected_file: str):
    """Refresh attack-related visualizations based on selected attack file."""
    df = load_results_data(selected_file)

    if df is None:
        empty_msg = "No data available. Select a valid attack results file."
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        empty_img = fig_to_image(fig)
        return empty_msg, empty_img, empty_img, empty_img, empty_img

    summary = generate_summary(df, selected_file)
    model_chart = create_model_asr_chart(df)
    attack_chart = create_attack_category_chart(df)
    heatmap = create_heatmap(df)
    pie_chart = create_results_breakdown_pie(df)

    return summary, model_chart, attack_chart, heatmap, pie_chart


def refresh_defense_visualizations(selected_file: str):
    """Refresh defense-related visualizations based on selected defense file."""
    df = load_results_data(selected_file)

    if df is None:
        empty_msg = "No data available. Select a valid defense results file."
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        empty_img = fig_to_image(fig)
        return empty_msg, empty_img, empty_img, empty_img, empty_img, empty_img

    summary = generate_summary(df, selected_file)
    model_chart = create_model_asr_chart(df)
    defense_effectiveness = create_defense_effectiveness_chart(df)
    defense_comparison = create_defense_asr_comparison(df)
    heatmap = create_heatmap(df)
    pie_chart = create_results_breakdown_pie(df)

    return summary, model_chart, defense_effectiveness, defense_comparison, heatmap, pie_chart


def create_visualization_tab():
    """Create the Visualization tab UI component."""

    with gr.TabItem("Visualization", id="viz"):
        # Attack Results Section
        gr.Markdown("### Attack Testing Results")

        with gr.Row():
            attack_file_dropdown = gr.Dropdown(
                choices=get_attack_file_choices(),
                value=get_attack_file_choices()[0] if get_attack_file_choices(
                ) and get_attack_file_choices()[0] != "No attack results files found" else None,
                label="Select Attack Results File",
                interactive=True
            )
            attack_refresh_btn = gr.Button(
                "Refresh", variant="primary", scale=0)

        with gr.Row():
            attack_summary_md = gr.Markdown(
                "Select an attack results file to view charts.")

        with gr.Row():
            attack_model_chart = gr.Image(label="ASR by Model")
            attack_category_chart = gr.Image(label="ASR by Attack Category")

        with gr.Row():
            attack_heatmap = gr.Image(label="Model x Attack Heatmap")
            attack_pie_chart = gr.Image(label="Results Breakdown")

        # Defense Results Section
        gr.Markdown("---")
        gr.Markdown("### Defense Testing Results")

        with gr.Row():
            defense_file_dropdown = gr.Dropdown(
                choices=get_defense_file_choices(),
                value=get_defense_file_choices()[0] if get_defense_file_choices(
                ) and get_defense_file_choices()[0] != "No defense results files found" else None,
                label="Select Defense Results File",
                interactive=True
            )
            defense_refresh_btn = gr.Button(
                "Refresh", variant="primary", scale=0)

        with gr.Row():
            defense_summary_md = gr.Markdown(
                "Select a defense results file to view charts.")

        with gr.Row():
            defense_model_chart = gr.Image(label="ASR by Model (with Defense)")
            defense_effectiveness_chart = gr.Image(
                label="Defense Effectiveness (DE = 1 - ASR_defended/ASR_baseline)")

        with gr.Row():
            defense_comparison_chart = gr.Image(
                label="ASR Comparison Across Strategies")
            defense_heatmap = gr.Image(label="Model x Attack Heatmap")

        with gr.Row():
            defense_pie_chart = gr.Image(label="Results Breakdown")

        # Event handlers for attack section
        attack_file_dropdown.change(
            fn=refresh_attack_visualizations,
            inputs=[attack_file_dropdown],
            outputs=[attack_summary_md, attack_model_chart,
                     attack_category_chart, attack_heatmap, attack_pie_chart]
        )

        attack_refresh_btn.click(
            fn=refresh_attack_visualizations,
            inputs=[attack_file_dropdown],
            outputs=[attack_summary_md, attack_model_chart,
                     attack_category_chart, attack_heatmap, attack_pie_chart]
        )

        # Event handlers for defense section
        defense_file_dropdown.change(
            fn=refresh_defense_visualizations,
            inputs=[defense_file_dropdown],
            outputs=[defense_summary_md, defense_model_chart, defense_effectiveness_chart,
                     defense_comparison_chart, defense_heatmap, defense_pie_chart]
        )

        defense_refresh_btn.click(
            fn=refresh_defense_visualizations,
            inputs=[defense_file_dropdown],
            outputs=[defense_summary_md, defense_model_chart, defense_effectiveness_chart,
                     defense_comparison_chart, defense_heatmap, defense_pie_chart]
        )
