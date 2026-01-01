import logging
from typing import List

import gradio as gr

from src.config_loader import get_config_loader, get_attack_categories
from src.inference import load_model, generate_response
from src.detection import detect_attack
from src.metrics import get_metrics_collector
from src.defense_prompts import (
    get_defense_builder, set_defense_strategy,
    get_all_defense_configs, get_defense_description, DefenseStrategy
)

logger = logging.getLogger(__name__)


def run_batch_test(
    selected_models: List[str],
    selected_categories: List[str],
    defense_strategy: str,
    app_state
):
    """
    Run batch tests across selected models and attack categories.
    Uses generator to provide live logging updates.

    Args:
        selected_models: List of model keys to test
        selected_categories: List of attack category keys to test
        defense_strategy: Defense strategy to apply (e.g., 'none', 'sandwich')
        app_state: Application state object

    Yields:
        Tuple of (status_log, progress_text, comparison_df) with live updates
    """
    if not selected_models:
        yield "[WARNING] Please select at least one model", "Ready", None
        return

    if not selected_categories:
        yield "[WARNING] Please select at least one attack category", "Ready", None
        return

    if app_state.is_batch_running:
        yield "[WARNING] A batch test is already running", "Busy", None
        return

    # Set defense strategy
    try:
        strategy = DefenseStrategy(defense_strategy)
        set_defense_strategy(strategy)
    except ValueError:
        set_defense_strategy(DefenseStrategy.NONE)

    defense_config = get_defense_builder().get_config()

    app_state.set_batch_running(True)

    try:
        # Clear previous results
        collector = get_metrics_collector()
        collector.clear()

        # Get attacks for selected categories
        all_categories = get_attack_categories()
        attacks_to_run = []

        for cat in all_categories:
            if cat.key in selected_categories:
                for var in cat.variations:
                    attacks_to_run.append((cat, var))

        total_tests = len(selected_models) * len(attacks_to_run)
        completed = 0

        log_lines = [
            "# Attack Batch Testing Started",
            f"- **Defense Strategy:** {defense_config.name}",
            f"- Models: {len(selected_models)}",
            f"- Attack Categories: {len(selected_categories)}",
            f"- Total Tests: {total_tests}", ""
        ]

        # Initial yield to show test is starting
        yield "\n".join(log_lines), f"Starting... 0/{total_tests}", None

        # Run tests
        for model_key in selected_models:
            # Load model
            log_lines.append(f"## Loading model: {model_key}")
            yield "\n".join(log_lines), f"Loading model... {completed}/{total_tests}", None

            success, msg = load_model(model_key)
            if not success:
                log_lines.append(f"[FAILED] Failed to load model: {msg}")
                yield "\n".join(log_lines), f"Model failed {completed}/{total_tests}", None
                continue

            model_config = get_config_loader().get_model(model_key)
            log_lines.append(f"[OK] Model loaded: {model_config.name}")
            yield "\n".join(log_lines), f"Model loaded {completed}/{total_tests}", None

            # Run attacks
            for cat, attack in attacks_to_run:
                # Show prompt being sent
                log_lines.append(
                    f"\n### Test {completed + 1}/{total_tests}: {attack.name}")
                log_lines.append(f"**Category:** {cat.category}")

                # Escape backticks to prevent breaking code block formatting
                prompt_preview = attack.prompt[:500] + \
                    ('...' if len(attack.prompt) > 500 else '')
                prompt_preview = prompt_preview.replace('```', '\\`\\`\\`')
                log_lines.append(f"\n**Prompt:**\n```\n{prompt_preview}\n```")
                yield "\n".join(log_lines), f"Running test {completed + 1}/{total_tests}: {attack.name[:30]}...", None

                try:
                    # Generate response with attack object for document injection
                    inference_result = generate_response(
                        attack.prompt, attack=attack)

                    if inference_result and inference_result.success:
                        # Detect success
                        detection_result = detect_attack(
                            attack,
                            inference_result.response,
                            cat.category
                        )

                        # Store result
                        collector.add_result(
                            model_config=model_config,
                            attack=attack,
                            attack_category=cat.category,
                            inference_result=inference_result,
                            detection_result=detection_result
                        )

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
                    elif inference_result:
                        log_lines.append(
                            f"\n**Response:** [ERROR] {inference_result.error_message}")
                        log_lines.append("---")
                    else:
                        log_lines.append(
                            f"\n**Response:** [ERROR] No response from inference engine")
                        log_lines.append("---")

                except Exception as e:
                    logger.exception(
                        f"Error running attack {attack.name}: {e}")
                    log_lines.append(f"\n**Response:** [FAILED] {str(e)}")
                    log_lines.append("---")

                completed += 1
                # Live update after each attack
                pct = int((completed / total_tests) * 100)
                yield "\n".join(log_lines), f"Progress: {completed}/{total_tests} ({pct}%)", None

        # Generate summary
        summary = collector.get_summary()
        model_metrics = collector.get_model_metrics()
        attack_metrics = collector.get_attack_metrics()

        # Create comparison table data
        import pandas as pd
        table_data = []
        for key, mm in model_metrics.items():
            successful_attacks = mm.total_tests - mm.blocked_attacks
            blocked_attacks = mm.blocked_attacks

            table_data.append({
                'model': mm.model_name,
                'asr': mm.asr * 100,
                'successful_attacks': f"{successful_attacks}/{mm.total_tests}",
                'blocked_attacks': f"{blocked_attacks}/{mm.total_tests}",
                'defense_strategy': defense_config.name,
            })

        comparison_df = pd.DataFrame(table_data)
        comparison_df.columns = [
            'Model', 'ASR (%)', 'Attacks Succeeded', 'Blocked Attacks', 'Defense Strategy']
        comparison_df['ASR (%)'] = comparison_df['ASR (%)'].round(1)

        # Export to CSV
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = collector.export_to_csv(f"attack_results_{timestamp}.csv")

        log_lines.append("")
        log_lines.append(f"# [COMPLETE] Batch test complete!")
        log_lines.append(f"Results exported to: {csv_path}")

        yield "\n".join(log_lines), "[COMPLETE] Batch test finished!", comparison_df

    finally:
        app_state.set_batch_running(False)


def create_attack_testing_tab(model_choices, category_choices, app_state):
    """Create the Attack Testing tab UI component."""

    with gr.TabItem("Attack Testing", id="batch"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                ### Automated Attack Testing
                Run all selected attacks against all selected models automatically.
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

                defense_dropdown = gr.Dropdown(
                    choices=get_all_defense_configs(),
                    value="none",
                    label="Defense Strategy",
                )
                defense_info = gr.Markdown(get_defense_description("none"))

                batch_btn = gr.Button(
                    "Start Attack Testing", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### Progress")
                progress_text = gr.Markdown("", elem_id="progress-text")

                with gr.Accordion("Test Log", open=True):
                    with gr.Column(elem_id="batch-log-wrapper"):
                        batch_log = gr.Markdown(value="Ready to start...")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Comparison Table")
                batch_summary = gr.Dataframe(
                    headers=["Model", "ASR (%)", "Attacks Succeeded",
                             "Blocked Attacks", "Defense Strategy"],
                    interactive=False
                )

        # Update defense info when dropdown changes
        defense_dropdown.change(
            fn=get_defense_description,
            inputs=[defense_dropdown],
            outputs=[defense_info]
        )

        # Convert display names back to keys for batch testing
        def batch_test_wrapper(models_display, categories_display, defense_strategy):
            # Convert display names to keys
            model_key_map = {m[0]: m[1] for m in model_choices}
            category_key_map = {c[0]: c[1] for c in category_choices}

            model_keys = [model_key_map[m]
                          for m in models_display if m in model_key_map]
            category_keys = [category_key_map[c]
                             for c in categories_display if c in category_key_map]

            # Yield from the generator to pass through live updates
            yield from run_batch_test(model_keys, category_keys, defense_strategy, app_state)

        batch_btn.click(
            fn=batch_test_wrapper,
            inputs=[model_checkboxes, category_checkboxes, defense_dropdown],
            outputs=[batch_log, progress_text, batch_summary],
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
                }, 100);
            }
            """
        )
