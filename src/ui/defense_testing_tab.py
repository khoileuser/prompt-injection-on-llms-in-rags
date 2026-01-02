import logging
from typing import List
import pandas as pd

import gradio as gr

from src.config_loader import get_config_loader, get_attack_categories
from src.inference import load_model, generate_response
from src.detection import detect_attack
from src.metrics import get_metrics_collector
from src.defense_prompts import (
    DefenseStrategy, set_defense_strategy, get_defense_builder,
)

logger = logging.getLogger(__name__)


def run_comparative_batch_test(
    selected_models: List[str],
    selected_categories: List[str],
    app_state
):
    """
    Run batch tests across ALL defense strategies for comparative analysis.

    This automatically runs tests with:
    1. No Defense (baseline)
    2. Reminder Defense
    3. Sandwich Defense
    4. Instructional Defense
    5. Spotlighting Defense
    6. Isolation Defense

    Args:
        selected_models: List of model keys to test
        selected_categories: List of attack category keys to test
        app_state: Application state object

    Yields:
        Tuple of (progress_text, status_log, comparison_df) with live updates
    """
    if not selected_models:
        yield "Ready", "[WARNING] Please select at least one model", None
        return

    if not selected_categories:
        yield "Ready", "[WARNING] Please select at least one attack category", None
        return

    if app_state.is_batch_running:
        yield "Busy", "[WARNING] A batch test is already running", None
        return

    app_state.set_batch_running(True)

    try:
        # Get attacks for selected categories
        all_categories = get_attack_categories()
        attacks_to_run = []

        for cat in all_categories:
            if cat.key in selected_categories:
                for var in cat.variations:
                    attacks_to_run.append((cat, var))

        # Calculate total tests across all defense strategies
        all_strategies = list(DefenseStrategy)
        total_tests = len(selected_models) * \
            len(attacks_to_run) * len(all_strategies)
        completed = 0

        log_lines = [
            "# Defense Batch Testing Started",
            f"- **Testing ALL {len(all_strategies)} defense strategies**",
            f"- Models: {len(selected_models)}",
            f"- Attack Categories: {len(selected_categories)}",
            f"- Attacks per strategy: {len(attacks_to_run)}",
            f"- **Total Tests: {total_tests}**", ""
        ]

        # Storage for comparative results
        comparative_results = []

        # Initial yield
        yield f"Starting... 0/{total_tests}", "\n".join(log_lines), None

        # ITERATE THROUGH EACH DEFENSE STRATEGY
        for strategy_idx, defense_strategy in enumerate(all_strategies, 1):
            set_defense_strategy(defense_strategy)
            defense_config = get_defense_builder().get_config()

            log_lines.append(
                f"## Strategy {strategy_idx}/{len(all_strategies)}: {defense_config.name}")
            log_lines.append(f"*{defense_config.description}*\n")
            yield f"Strategy {strategy_idx}/{len(all_strategies)}: {defense_strategy.value}", "\n".join(log_lines), None

            # Track results for this strategy
            strategy_success_count = 0
            strategy_total_count = 0

            # Run tests for each model
            for model_key in selected_models:
                log_lines.append(f"\n### Loading model: {model_key}")
                yield f"Loading model... {completed}/{total_tests}", "\n".join(log_lines), None

                success, msg = load_model(model_key)
                if not success:
                    log_lines.append(f"[FAILED] Failed to load model: {msg}")
                    yield f"Model failed {completed}/{total_tests}", "\n".join(log_lines), None
                    continue

                model_config = get_config_loader().get_model(model_key)
                log_lines.append(f"[OK] Loaded: {model_config.name}")

                # Track model-specific results
                model_success_count = 0
                model_total_count = 0

                # Run attacks
                for cat, attack in attacks_to_run:
                    try:
                        # Log current test
                        log_lines.append(
                            f"\n### Test {completed + 1}/{total_tests}: {attack.name}")
                        log_lines.append(f"**Category:** {cat.category}")

                        # Show prompt being sent
                        prompt_preview = attack.prompt[:500] + \
                            ('...' if len(attack.prompt) > 500 else '')
                        prompt_preview = prompt_preview.replace(
                            '```', '\\`\\`\\`')
                        log_lines.append(
                            f"\n**Prompt:**\n```\n{prompt_preview}\n```")

                        # Generate response with current defense strategy and attack object
                        inference_result = generate_response(
                            attack.prompt, attack=attack)

                        if inference_result and inference_result.success:
                            # Detect success
                            detection_result = detect_attack(
                                attack,
                                inference_result.response,
                                cat.category
                            )

                            # ADD TO METRICS COLLECTOR for visualization
                            collector = get_metrics_collector()
                            collector.add_result(
                                model_config=model_config,
                                attack=attack,
                                attack_category=cat.category,
                                inference_result=inference_result,
                                detection_result=detection_result,
                                defense_strategy=defense_strategy.value  # Pass the strategy explicitly
                            )

                            # Track statistics
                            is_success = detection_result.result.value == "success"
                            if is_success:
                                strategy_success_count += 1
                                model_success_count += 1

                            strategy_total_count += 1
                            model_total_count += 1

                            # Log result
                            result_label = {
                                "success": "[SUCCESS]",
                                "blocked": "[BLOCKED]",
                                "error": "[ERROR]"
                            }.get(detection_result.result.value, "[UNKNOWN]")

                            # Show response from LLM
                            response_preview = inference_result.response[:800] + (
                                '...' if len(inference_result.response) > 800 else '')
                            # Escape backticks to prevent breaking code block formatting
                            response_preview = response_preview.replace(
                                '```', '\\`\\`\\`')
                            log_lines.append(
                                f"\n**Response:**\n```\n{response_preview}\n```")
                            log_lines.append(
                                f"\n**Result:** {result_label} | **Time:** {inference_result.response_time:.2f}s")
                            log_lines.append("---")

                            # Store for comparative analysis
                            comparative_results.append({
                                'defense_strategy': defense_config.name,
                                'defense_value': defense_strategy.value,
                                'model': model_config.name,
                                'model_key': model_key,
                                'attack_id': attack.id,
                                'attack_name': attack.name,
                                'attack_category': cat.category,
                                'result': detection_result.result.value,
                                'success': is_success,
                                'response_time': inference_result.response_time
                            })

                        elif inference_result:
                            log_lines.append(
                                f"\n**Response:** [ERROR] Inference failed: {inference_result.error_message}")
                            log_lines.append("---")
                            model_total_count += 1
                            strategy_total_count += 1
                        else:
                            log_lines.append(
                                f"\n**Response:** [ERROR] No response from inference engine")
                            log_lines.append("---")
                            model_total_count += 1
                            strategy_total_count += 1

                    except Exception as e:
                        logger.exception(
                            f"Error running attack {attack.name}: {e}")
                        log_lines.append(f"\n**Response:** [FAILED] {str(e)}")
                        log_lines.append("---")
                        model_total_count += 1
                        strategy_total_count += 1

                    completed += 1
                    pct = int((completed / total_tests) * 100)

                    # Update progress every test (live logging)
                    yield f"Progress: {completed}/{total_tests} ({pct}%)", "\n".join(log_lines), None

                # Show model summary
                model_asr = (model_success_count / model_total_count *
                             100) if model_total_count > 0 else 0
                log_lines.append(
                    f"   {model_config.name} ASR: **{model_asr:.1f}%** ({model_success_count}/{model_total_count} attacks succeeded)")

            # Show strategy summary
            strategy_asr = (strategy_success_count / strategy_total_count *
                            100) if strategy_total_count > 0 else 0
            log_lines.append(
                f"\n### {defense_config.name} Overall ASR: **{strategy_asr:.1f}%**")
            log_lines.append(
                f"   ({strategy_success_count}/{strategy_total_count} total attacks succeeded)\n")

            yield f"Completed {strategy_idx}/{len(all_strategies)} strategies", "\n".join(log_lines), None

        # Generate comparative analysis
        log_lines.append("")
        log_lines.append(f"# [COMPLETE] Batch test complete!")

        # Create DataFrame for analysis
        df = pd.DataFrame(comparative_results)

        # Calculate ASR per defense strategy
        asr_by_defense = df.groupby('defense_strategy')['success'].mean() * 100
        asr_by_defense = asr_by_defense.sort_values(ascending=False)

        log_lines.append(
            "\n### Attack Success Rate (ASR) by Defense Strategy:")
        log_lines.append("*(Lower is better)*\n")

        baseline_asr = None
        for defense_name, asr in asr_by_defense.items():
            if 'Baseline' in defense_name:
                baseline_asr = asr
                log_lines.append(
                    f"- **{defense_name}**: {asr:.1f}% (Baseline)")
            else:
                reduction = ""
                if baseline_asr:
                    pct_reduction = (
                        (baseline_asr - asr) / baseline_asr * 100) if baseline_asr > 0 else 0
                    reduction = f" (↓{pct_reduction:.1f}% reduction)"
                log_lines.append(
                    f"- **{defense_name}**: {asr:.1f}%{reduction}")

        # Calculate ASR per model
        log_lines.append("\n### Attack Success Rate by Model:")
        asr_by_model = df.groupby(['model', 'defense_strategy'])[
            'success'].mean() * 100

        for model in df['model'].unique():
            log_lines.append(f"\n**{model}:**")
            model_data = asr_by_model[model].sort_values(ascending=False)
            for defense, asr in model_data.items():
                log_lines.append(f"  - {defense}: {asr:.1f}%")

        # Export results
        log_lines.append("\n### Exporting Results...")

        from pathlib import Path
        from datetime import datetime

        output_dir = Path(__file__).parent.parent.parent / "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export via MetricsCollector for detailed results
        collector = get_metrics_collector()
        csv_path = collector.export_to_csv(f"defense_results_{timestamp}.csv")
        log_lines.append(
            f"[OK] Results exported: `defense_results_{timestamp}.csv`")

        # Create comparison table for UI from existing data
        summary_data = []
        for defense in asr_by_defense.index:
            defense_df = df[df['defense_strategy'] == defense]

            # Create a row for each model within this defense strategy
            for model in defense_df['model'].unique():
                model_df = defense_df[defense_df['model'] == model]
                total_tests = len(model_df)
                successful_attacks = int(model_df['success'].sum())
                blocked_attacks = int((model_df['result'] == 'blocked').sum())
                model_asr = (successful_attacks / total_tests *
                             100) if total_tests > 0 else 0

                summary_data.append({
                    'defense_strategy': defense,
                    'model': model,
                    'overall_asr': model_asr,
                    'successful_attacks': f"{successful_attacks}/{total_tests}",
                    'blocked_attacks': f"{blocked_attacks}/{total_tests}",
                    'avg_response_time': model_df['response_time'].mean(),
                })

        summary_df = pd.DataFrame(summary_data)

        # Create comparison table for UI
        comparison_table = summary_df[['defense_strategy', 'model', 'overall_asr',
                                       'successful_attacks', 'blocked_attacks']].copy()
        comparison_table.columns = ['Defense Strategy', 'Model',
                                    'ASR (%)', 'Attacks Succeeded', 'Blocked Attacks']
        comparison_table['ASR (%)'] = comparison_table['ASR (%)'].round(1)

        log_lines.append(f"\n**Defense testing complete!**")
        log_lines.append(f"Total tests run: {total_tests}")

        yield "COMPLETE", "\n".join(log_lines), comparison_table

    except Exception as e:
        logger.error(f"Error in defense batch test: {e}")
        yield "Error", f"[ERROR] {str(e)}", None
    finally:
        app_state.set_batch_running(False)


def create_defense_testing_tab(model_choices, category_choices, app_state):
    """Create the Defense Testing tab UI component."""

    with gr.TabItem("Defense Testing", id="comparative"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                ### Automated Defense Testing
                Automatically tests all defense strategies against selected models and attack categories.
                """)

                gr.Markdown("#### Configuration")

                model_checkboxes = gr.CheckboxGroup(
                    choices=[m[0] for m in model_choices],
                    value=[m[0] for m in model_choices],
                    label="Models to Test",
                )

                category_checkboxes = gr.CheckboxGroup(
                    choices=[c[0] for c in category_choices],
                    value=[c[0] for c in category_choices],
                    label="Attack Categories",
                )

                gr.Markdown("""
                **Defense Strategies to Test:**
                - No Defense (Baseline)
                - Reminder Defense
                - Sandwich Defense
                - Instructional Hierarchy
                - Spotlighting
                - Strict Isolation
                """)

                comparative_btn = gr.Button(
                    "Start Defense Testing", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("#### Progress")
                progress_text = gr.Markdown("", elem_id="progress-text")

                with gr.Accordion("Test Log", open=True):
                    with gr.Column(elem_id="batch-log-wrapper"):
                        batch_log = gr.Markdown(value="Ready to start...")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Comparison Table")
                comparison_table_ui = gr.Dataframe(
                    headers=["Defense Strategy", "Model",
                             "ASR (%)", "Attacks Succeeded", "Blocked Attacks"],
                    interactive=False
                )

        # Convert display names back to keys
        def comparative_test_wrapper(models_display, categories_display):
            model_key_map = {m[0]: m[1] for m in model_choices}
            category_key_map = {c[0]: c[1] for c in category_choices}

            model_keys = [model_key_map[m]
                          for m in models_display if m in model_key_map]
            category_keys = [category_key_map[c]
                             for c in categories_display if c in category_key_map]

            yield from run_comparative_batch_test(model_keys, category_keys, app_state)

        comparative_btn.click(
            fn=comparative_test_wrapper,
            inputs=[model_checkboxes, category_checkboxes],
            outputs=[progress_text, batch_log, comparison_table_ui],
            show_progress="hidden"
        )

        # Add auto-scroll on batch_log changes
        batch_log.change(
            fn=None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                setTimeout(() => {
                    const wrapper = document.getElementById('batch-log-wrapper');
                    if (wrapper) {
                        wrapper.scrollTop = wrapper.scrollHeight;
                    }
                }, 200);
                return [];
            }
            """
        )
