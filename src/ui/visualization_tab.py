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

    total_tests = len(df)
    unique_models = df['model_name'].nunique()
    unique_attacks = df['attack_category'].nunique()
    unique_defenses = df['defense_strategy'].nunique()
    overall_asr = calculate_asr(df) * 100

    model_asr = {}
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        model_asr[model] = calculate_asr(model_df) * 100

    summary = f"""
| Metric | Value |
|--------|-------|
| Total Tests | {total_tests} |
| Models Tested | {unique_models} |
| Attack Categories | {unique_attacks} |
| Defense Strategies | {unique_defenses} |
| Overall ASR | {overall_asr:.1f}% |
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
    """Create bar chart for ASR by attack objective."""
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
    ax.set_title('ASR by Attack Objective', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    for bar, asr in zip(bars, attack_df['ASR']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{asr:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    return fig_to_image(fig)


def create_matrix_heatmap(df: pd.DataFrame):
    """Create 2×3 matrix heatmap: Injection Vector × Attack Objective."""
    if df is None or df.empty:
        return None

    # Try to get injection_vector and attack_objective from columns
    # If not present, derive from attack_category
    if 'injection_vector' not in df.columns or 'attack_objective' not in df.columns:
        logger.info(
            "Deriving injection_vector and attack_objective from attack_category")

        # Parse attack_category to extract vector and objective
        # Format is typically: "Direct Instruction Override" or "direct_instruction_override"
        def parse_category(category):
            category = str(category).lower()

            # Determine injection vector - check indirect first to avoid substring match
            if 'indirect' in category:
                vector = 'indirect'
            elif 'direct' in category:
                vector = 'direct'
            else:
                vector = 'unknown'

            # Determine attack objective
            if 'instruction' in category or 'override' in category:
                objective = 'instruction_override'
            elif 'data' in category or 'extraction' in category:
                objective = 'data_extraction'
            elif 'role' in category or 'confusion' in category:
                objective = 'role_confusion'
            else:
                objective = 'unknown'

            return vector, objective

        # Create new columns
        df = df.copy()
        df[['injection_vector', 'attack_objective']] = df['attack_category'].apply(
            lambda x: pd.Series(parse_category(x))
        )

    # Get unique vectors and objectives - maintain specific order
    # Force the order: direct first, then indirect
    all_vectors = [v for v in df['injection_vector'].unique()
                   if v != 'unknown']
    vectors = []
    if 'direct' in all_vectors:
        vectors.append('direct')
    if 'indirect' in all_vectors:
        vectors.append('indirect')

    # Sort objectives alphabetically
    objectives = sorted(
        [o for o in df['attack_objective'].unique() if o != 'unknown'])

    if not vectors or not objectives:
        logger.warning(
            f"Insufficient data for matrix: vectors={vectors}, objectives={objectives}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'Insufficient data for 2×3 matrix\nRequired: injection_vector and attack_objective data',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        return fig_to_image(fig)

    # Build matrix
    matrix = []
    for vector in vectors:
        row = []
        for objective in objectives:
            subset = df[(df['injection_vector'] == vector) &
                        (df['attack_objective'] == objective)]
            asr = calculate_asr(subset) * 100
            row.append(asr)
        matrix.append(row)

    # Format labels
    vector_labels = [v.capitalize() for v in vectors]
    objective_labels = [o.replace('_', ' ').title() for o in objectives]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    # Use same colormap as ASR heatmap: RdYlGn_r (Red-Yellow-Green reversed)
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels(objective_labels, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(len(vectors)))
    ax.set_yticklabels(vector_labels, rotation=0, fontsize=12)

    # Add text annotations
    for i in range(len(vectors)):
        for j in range(len(objectives)):
            text = ax.text(j, i, f'{matrix[i][j]:.1f}%', ha='center', va='center',
                           color='white' if matrix[i][j] > 50 else 'black',
                           fontsize=14, fontweight='bold')

    ax.set_xlabel('Attack Objective', fontsize=14, fontweight='bold')
    ax.set_ylabel('Injection Vector', fontsize=14, fontweight='bold')
    ax.set_title('Attack Success Rate',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attack Success Rate (%)')
    cbar.ax.tick_params(labelsize=10)

    # Add success criteria annotation
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

    ax.set_title('ASR Heatmap on Attack Objectives',
                 fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='ASR (%)')
    plt.tight_layout()
    return fig_to_image(fig)


def create_defense_effectiveness_chart(df: pd.DataFrame):
    """Create grouped bar chart for Defense Effectiveness."""
    if df is None or df.empty:
        return None

    strategies = df['defense_strategy'].unique().tolist()

    # If 'none' is not in the current dataframe, try to load it from attack results
    if 'none' not in strategies:
        logger.info(
            "No baseline 'none' strategy found in current data. Attempting to load from attack results...")

        # Try to find the most recent attack results file
        attack_files = get_attack_file_choices()
        if attack_files and attack_files[0] != "No attack results files found":
            baseline_df = load_results_data(attack_files[0])

            if baseline_df is not None and 'none' in baseline_df['defense_strategy'].unique():
                logger.info(
                    f"Found baseline data in {attack_files[0]}. Merging with defense results...")
                # Merge the baseline data with the defense data
                df = pd.concat([baseline_df, df], ignore_index=True)
                strategies = df['defense_strategy'].unique().tolist()
                logger.info(f"Merged data. Strategies now: {strategies}")
            else:
                logger.warning(
                    "Could not find baseline data with 'none' strategy in attack results")

    # Check if we have enough data for comparison
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
    ax.set_title('Defense Effectiveness by Mechanism',
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

    # If we don't have baseline data, try to load it from attack results
    if 'none' not in strategies:
        logger.info(
            "No baseline 'none' strategy found for comparison. Attempting to load from attack results...")

        attack_files = get_attack_file_choices()
        if attack_files and attack_files[0] != "No attack results files found":
            baseline_df = load_results_data(attack_files[0])

            if baseline_df is not None and 'none' in baseline_df['defense_strategy'].unique():
                logger.info(
                    f"Found baseline data in {attack_files[0]}. Merging with defense results...")
                df = pd.concat([baseline_df, df], ignore_index=True)
                strategies = df['defense_strategy'].unique().tolist()
                logger.info(f"Merged data. Strategies now: {strategies}")

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

    fig, ax = plt.subplots(figsize=(4, 6))
    wedges, texts, autotexts = ax.pie(
        result_counts.values,
        labels=[r.capitalize() for r in result_counts.index],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax.set_title('Results Summary',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig_to_image(fig)


def create_defense_vs_attack_heatmap(df: pd.DataFrame):
    """Create heatmap showing Defense Strategy effectiveness against Attack Objectives."""
    if df is None or df.empty:
        return None

    strategies = df['defense_strategy'].unique().tolist()

    # If 'none' is not in the current dataframe, try to load it from attack results
    if 'none' not in strategies:
        logger.info(
            "No baseline 'none' strategy found. Attempting to load from attack results...")

        attack_files = get_attack_file_choices()
        if attack_files and attack_files[0] != "No attack results files found":
            baseline_df = load_results_data(attack_files[0])

            if baseline_df is not None and 'none' in baseline_df['defense_strategy'].unique():
                logger.info(
                    f"Found baseline data in {attack_files[0]}. Merging with defense results...")
                df = pd.concat([baseline_df, df], ignore_index=True)
                strategies = df['defense_strategy'].unique().tolist()
                logger.info(f"Merged data. Strategies now: {strategies}")

    # Check if we have enough data
    if len(strategies) <= 1 or 'none' not in strategies:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Defense vs Attack analysis requires baseline and defense data.\nRun Attack and Defense Testing first.',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Defense Mechanisms against Attack Objectives',
                     fontsize=14, fontweight='bold')
        return fig_to_image(fig)

    defense_strategies = [s for s in strategies if s != 'none']
    attack_categories = df['attack_category'].unique().tolist()

    # Build matrix showing ASR% for all rows (baseline + defense strategies)
    # Lower ASR% = Better (fewer successful attacks)
    matrix = []

    # Add baseline row (showing ASR with no defense)
    baseline_row = []
    for category in attack_categories:
        baseline_subset = df[(df['defense_strategy'] == 'none') &
                             (df['attack_category'] == category)]
        baseline_asr = calculate_asr(
            baseline_subset) * 100  # Convert to percentage
        baseline_row.append(baseline_asr)
    matrix.append(baseline_row)

    # Add defense strategy ASR rows
    for strategy in defense_strategies:
        row = []
        for category in attack_categories:
            # Get defended ASR for this strategy and category
            defended_subset = df[(df['defense_strategy'] == strategy) &
                                 (df['attack_category'] == category)]
            defended_asr = calculate_asr(
                defended_subset) * 100  # Convert to percentage

            row.append(defended_asr)
        matrix.append(row)

    # Shorten strategy names for display - include "None (Baseline)" at top
    # Shorten strategy names for display - include "None (Baseline)" at top
    short_strategies = [
        'None (Baseline)'] + [s.replace('_', ' ').title() for s in defense_strategies]
    short_strategies = [
        'None (Baseline)'] + [s.replace('_', ' ').title() for s in defense_strategies]

    # Shorten attack category names
    short_categories = []
    for cat in attack_categories:
        # Remove "Direct" or "Indirect" prefix and shorten
        cat_parts = cat.split()
        if len(cat_parts) > 2:
            short_cat = ' '.join(cat_parts[:2])
        else:
            short_cat = cat
        short_categories.append(short_cat)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Use different colormaps for baseline row (ASR) vs effectiveness rows
    # For baseline row: higher ASR = worse (red), lower ASR = better (green)
    # For effectiveness rows: higher % = better (green), lower % = worse (red)
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(short_categories)))
    ax.set_xticklabels(short_categories, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(short_strategies)))
    ax.set_yticklabels(short_strategies, fontsize=10)

    # Add text annotations - all cells show ASR%
    for i in range(len(short_strategies)):
        for j in range(len(attack_categories)):
            asr_value = matrix[i][j]

            # All rows show ASR% - lower is better
            text_label = f'{asr_value:.1f}%'
            # Use white text on dark/red colors (high ASR), black on light/green colors (low ASR)
            text_color = 'white' if asr_value > 50 else 'black'

            text = ax.text(j, i, text_label, ha='center', va='center',
                           color=text_color, fontsize=9, fontweight='bold')

    # Update title to reflect that all cells show ASR%
    ax.set_title('ASR on Defense Mechanisms against Attack Objectives',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='ASR (%)')
    cbar.ax.text(1.5, 10, 'Better', rotation=90, va='center', fontsize=9)
    cbar.ax.text(1.5, 90, 'Worse', rotation=90, va='center', fontsize=9)

    plt.tight_layout()
    return fig_to_image(fig)


def create_detailed_summary_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Create a detailed summary table similar to defense testing results.
    Shows Defense x Model x Vector x Objective breakdown.
    """
    if df is None or df.empty:
        return None

    # Check if this is defense results (has defense_strategy column)
    if 'defense_strategy' not in df.columns:
        return None

    summary_data = []

    # Group by defense strategy
    for defense in df['defense_strategy'].unique():
        defense_df = df[df['defense_strategy'] == defense]

        # Group by model
        for model in defense_df['model_name'].unique():
            model_df = defense_df[defense_df['model_name'] == model]

            # Group by attack category
            for category in model_df['attack_category'].unique():
                cat_df = model_df[model_df['attack_category'] == category]

                if len(cat_df) > 0:
                    cat_total = len(cat_df)
                    cat_success = int(
                        (cat_df['detection_result'] == 'success').sum())
                    cat_blocked = int(
                        (cat_df['detection_result'] == 'blocked').sum())
                    cat_asr = (cat_success / cat_total *
                               100) if cat_total > 0 else 0

                    # Determine vector from category - check indirect first to avoid substring match
                    category_lower = category.lower()
                    if "indirect" in category_lower:
                        vector = "Indirect"
                    elif "direct" in category_lower:
                        vector = "Direct"
                    else:
                        vector = "Unknown"

                    # Extract objective only (remove Direct/Indirect prefix)
                    objective = category.replace(
                        "Direct ", "").replace("Indirect ", "")

                    summary_data.append({
                        "Defense": defense,
                        "Model": model,
                        "Vector": vector,
                        "Objective": objective,
                        "Tests": cat_total,
                        "Success": cat_success,
                        "Blocked": cat_blocked,
                        "ASR (%)": round(cat_asr, 1)
                    })

    if not summary_data:
        return None

    result_df = pd.DataFrame(summary_data)

    # Sort by Defense, Model, Vector, Objective
    result_df = result_df.sort_values(
        ["Defense", "Model", "Vector", "Objective"])

    return result_df


def create_attack_summary_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Create a detailed summary table for attack testing results.
    Shows Model x Vector x Objective breakdown.
    """
    if df is None or df.empty:
        return None

    summary_data = []

    # Group by model
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]

        # Group by attack category
        for category in model_df['attack_category'].unique():
            cat_df = model_df[model_df['attack_category'] == category]

            if len(cat_df) > 0:
                cat_total = len(cat_df)
                cat_success = int(
                    (cat_df['detection_result'] == 'success').sum())
                cat_blocked = int(
                    (cat_df['detection_result'] == 'blocked').sum())
                cat_asr = (cat_success / cat_total *
                           100) if cat_total > 0 else 0

                # Determine vector from category - check indirect first to avoid substring match
                category_lower = category.lower()
                if "indirect" in category_lower:
                    vector = "Indirect"
                elif "direct" in category_lower:
                    vector = "Direct"
                else:
                    vector = "Unknown"

                # Extract objective only (remove Direct/Indirect prefix)
                objective = category.replace(
                    "Direct ", "").replace("Indirect ", "")

                summary_data.append({
                    "Model": model,
                    "Vector": vector,
                    "Objective": objective,
                    "Tests": cat_total,
                    "Success": cat_success,
                    "Blocked": cat_blocked,
                    "ASR (%)": round(cat_asr, 1)
                })

    if not summary_data:
        return None

    result_df = pd.DataFrame(summary_data)

    # Sort by Model, Vector, Objective
    result_df = result_df.sort_values(["Model", "Vector", "Objective"])

    return result_df


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
        return empty_msg, empty_img, empty_img, empty_img, empty_img, empty_img, None

    summary = generate_summary(df, selected_file)
    model_chart = create_model_asr_chart(df)
    attack_chart = create_attack_category_chart(df)
    matrix_chart = create_matrix_heatmap(df)
    heatmap = create_heatmap(df)
    pie_chart = create_results_breakdown_pie(df)
    summary_table = create_attack_summary_table(df)

    return summary, model_chart, attack_chart, matrix_chart, heatmap, pie_chart, summary_table


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
        return empty_msg, empty_img, empty_img, empty_img, empty_img, empty_img, empty_img, None

    summary = generate_summary(df, selected_file)
    model_chart = create_model_asr_chart(df)
    defense_effectiveness = create_defense_effectiveness_chart(df)
    defense_comparison = create_defense_asr_comparison(df)
    defense_vs_attack = create_defense_vs_attack_heatmap(df)
    matrix_chart = create_matrix_heatmap(df)
    heatmap = create_heatmap(df)
    pie_chart = create_results_breakdown_pie(df)
    summary_table = create_detailed_summary_table(df)

    return summary, model_chart, defense_effectiveness, defense_comparison, defense_vs_attack, matrix_chart, heatmap, pie_chart, summary_table


def create_visualization_tab():
    """Create the Visualization tab UI component."""

    with gr.TabItem("Visualization", id="viz"):
        # Attack Results Section
        gr.Markdown("### Attack Testing Results")

        with gr.Row():
            attack_file_dropdown = gr.Dropdown(
                choices=get_attack_file_choices(),
                value=get_attack_file_choices()[0] if get_attack_file_choices(
                ) and get_attack_file_choices()[0] != "No results files found" else None,
                label="Select Results File",
                interactive=True,
                scale=2
            )
            attack_reload_btn = gr.Button("Reload")
            attack_refresh_btn = gr.Button("Load", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                attack_summary_md = gr.Markdown(
                    "Select a results file to view charts.")
            with gr.Column(scale=1):
                attack_summary_table = gr.Dataframe(
                    headers=["Model", "Vector", "Objective",
                             "Tests", "Success", "Blocked", "ASR (%)"],
                    label="",
                    interactive=False,
                    wrap=True
                )

        with gr.Row():
            attack_matrix_chart = gr.Image(
                label="Injection Vector × Attack Objective", scale=2)

        with gr.Row():
            attack_model_chart = gr.Image(label="ASR by Model")
            attack_category_chart = gr.Image(label="ASR by Attack Objective")

        with gr.Row():
            attack_heatmap = gr.Image(label="ASR Heatmap")
            attack_pie_chart = gr.Image(label="Results Summary")

        # Defense Results Section
        gr.Markdown("---")
        gr.Markdown("### Defense Testing Results")

        with gr.Row():
            defense_file_dropdown = gr.Dropdown(
                choices=get_defense_file_choices(),
                value=get_defense_file_choices()[0] if get_defense_file_choices(
                ) and get_defense_file_choices()[0] != "No results files found" else None,
                label="Select Results File",
                interactive=True,
                scale=2
            )
            defense_reload_btn = gr.Button("Reload")
            defense_refresh_btn = gr.Button("Load", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                defense_summary_md = gr.Markdown(
                    "Select a results file to view charts.")
            with gr.Column(scale=1):
                defense_summary_table = gr.Dataframe(
                    headers=["Defense", "Model", "Vector", "Objective",
                             "Tests", "Success", "Blocked", "ASR (%)"],
                    label="",
                    interactive=False,
                    wrap=True
                )

        with gr.Row():
            defense_matrix_chart = gr.Image(
                label="Injection Vector × Attack Objective", scale=2)

        with gr.Row():
            defense_model_chart = gr.Image(label="ASR by Model (with Defense)")
            defense_effectiveness_chart = gr.Image(
                label="Defense Effectiveness")

        with gr.Row():
            defense_comparison_chart = gr.Image(
                label="ASR Comparison Across Defenses")
            defense_vs_attack_heatmap = gr.Image(
                label="Defense Mechanisms against Attack Objectives")

        with gr.Row():
            defense_heatmap = gr.Image(label="ASR Heatmap (with Defense)")
            defense_pie_chart = gr.Image(label="Results Summary")

        # Event handlers for attack section
        attack_file_dropdown.change(
            fn=refresh_attack_visualizations,
            inputs=[attack_file_dropdown],
            outputs=[attack_summary_md, attack_model_chart,
                     attack_category_chart, attack_matrix_chart, attack_heatmap, attack_pie_chart, attack_summary_table]
        )

        attack_reload_btn.click(
            fn=lambda: gr.Dropdown(choices=get_attack_file_choices(),
                                   value=get_attack_file_choices()[0] if get_attack_file_choices() and get_attack_file_choices()[0] != "No attack results files found" else None),
            inputs=None,
            outputs=[attack_file_dropdown]
        )

        attack_refresh_btn.click(
            fn=refresh_attack_visualizations,
            inputs=[attack_file_dropdown],
            outputs=[attack_summary_md, attack_model_chart,
                     attack_category_chart, attack_matrix_chart, attack_heatmap, attack_pie_chart, attack_summary_table]
        )

        # Event handlers for defense section
        defense_file_dropdown.change(
            fn=refresh_defense_visualizations,
            inputs=[defense_file_dropdown],
            outputs=[defense_summary_md, defense_model_chart, defense_effectiveness_chart,
                     defense_comparison_chart, defense_vs_attack_heatmap, defense_matrix_chart, defense_heatmap, defense_pie_chart, defense_summary_table]
        )

        defense_reload_btn.click(
            fn=lambda: gr.Dropdown(choices=get_defense_file_choices(),
                                   value=get_defense_file_choices()[0] if get_defense_file_choices() and get_defense_file_choices()[0] != "No defense results files found" else None),
            inputs=None,
            outputs=[defense_file_dropdown]
        )

        defense_refresh_btn.click(
            fn=refresh_defense_visualizations,
            inputs=[defense_file_dropdown],
            outputs=[defense_summary_md, defense_model_chart, defense_effectiveness_chart,
                     defense_comparison_chart, defense_vs_attack_heatmap, defense_matrix_chart, defense_heatmap, defense_pie_chart, defense_summary_table]
        )
