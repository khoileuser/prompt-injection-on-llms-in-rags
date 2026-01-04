import logging
from datetime import datetime
from typing import List
import pandas as pd

import gradio as gr

from src.config_loader import get_config_loader, get_attack_categories
from src.inference import load_model, generate_response
from src.detection import (
    add_pending_detection,
    get_pending_result,
    process_pending_detections,
    clear_pending_detections,
    get_pending_count,
)
from src.metrics import get_metrics_collector
from src.defense_prompts import set_defense_strategy, DefenseStrategy

logger = logging.getLogger(__name__)


def run_batch_test(
    selected_models: List[str],
    selected_direct_categories: List[str],
    selected_indirect_categories: List[str],
    app_state,
):
    """
    Run batch tests across selected models and attack categories.
    Uses generator to provide live logging updates.

    Detection is done in batch at the end using Gemini API to minimize API calls.

    Args:
        selected_models: List of model keys to test
        selected_direct_categories: List of direct attack category keys to test
        selected_indirect_categories: List of indirect attack category keys to test
        app_state: Application state object

    Yields:
        Tuple of (status_log, progress_text, comparison_df) with live updates
    """
    if not selected_models:
        yield "[WARNING] Please select at least one model", "Ready", None
        return

    # Combine selected categories
    selected_categories = selected_direct_categories + selected_indirect_categories

    if not selected_categories:
        yield "[WARNING] Please select at least one attack category", "Ready", None
        return

    if app_state.is_batch_running:
        yield "[WARNING] A batch test is already running", "Busy", None
        return

    # Use no defense strategy (baseline testing)
    set_defense_strategy(DefenseStrategy.NONE)

    app_state.set_batch_running(True)

    # Clear any pending detections from previous runs
    clear_pending_detections()

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

        # Store pending detection indices with their metadata for later processing
        # List of (idx, model_config, attack, cat, inference_result)
        pending_results = []

        log_lines = [
            "Attack Batch Testing",
            f"- Models: {len(selected_models)}",
            f"- Attack Objectives: {len(selected_categories)}",
            f"- Total Tests: {total_tests}",
        ]

        # Initial yield to show test is starting
        yield "\n".join(log_lines), f"Starting... 0/{total_tests}", None

        # Run all attacks and collect responses
        log_lines.append("Running attacks and collecting responses...")
        yield "\n".join(log_lines), f"Running attacks...", None

        for model_key in selected_models:
            # Load model
            log_lines.append(f"\nLoading model: {model_key}")
            yield "\n".join(log_lines), f"Loading model... {completed}/{total_tests}", None

            success, msg = load_model(model_key)
            if not success:
                log_lines.append(f"\n[FAILED] Failed to load model: {msg}")
                yield "\n".join(log_lines), f"Model failed {completed}/{total_tests}", None
                continue

            model_config = get_config_loader().get_model(model_key)
            log_lines.append(f"\n[OK] Model loaded: {model_config.name}")
            yield "\n".join(log_lines), f"Model loaded {completed}/{total_tests}", None

            # Run attacks
            for cat, attack in attacks_to_run:
                # Show prompt being sent
                log_lines.append("\n")
                log_lines.append(
                    f"\nTest {completed + 1}/{total_tests}: {attack.name}")
                log_lines.append(f"\nObjective: {cat.category}")

                # Escape backticks to prevent breaking code block formatting
                prompt_preview = attack.prompt
                prompt_preview = prompt_preview.replace("```", "\\`\\`\\`")
                log_lines.append(f"\nPrompt:\n```\n{prompt_preview}\n```")
                yield "\n".join(log_lines), f"Test {completed + 1}/{total_tests}: {attack.name[:30]}...", None

                try:
                    # Generate response with attack object for document injection
                    inference_result = generate_response(
                        attack.prompt, attack=attack)

                    if inference_result and inference_result.success:
                        # Add to pending detection queue (will be processed in batch later)
                        detection_idx = add_pending_detection(
                            attack, inference_result.response, cat.category
                        )

                        # Store for later processing
                        pending_results.append(
                            (detection_idx, model_config,
                             attack, cat, inference_result)
                        )

                        # Show response from LLM (detection will come later)
                        response_preview = inference_result.response

                        # Escape backticks to prevent breaking code block formatting
                        response_preview = response_preview.replace(
                            "```", "\\`\\`\\`")
                        log_lines.append(
                            f"\nResponse:\n```\n{response_preview}\n```")
                        log_lines.append(
                            f"\nResponse Time: {inference_result.response_time:.2f}s | Tokens Generated: {inference_result.tokens_generated}")
                    elif inference_result:
                        log_lines.append(
                            f"\nResponse: [ERROR] {inference_result.error_message}")
                    else:
                        log_lines.append(
                            f"\nResponse: [ERROR] No response from inference engine")

                except Exception as e:
                    logger.exception(
                        f"\nError running attack {attack.name}: {e}")
                    log_lines.append(f"\nResponse: [FAILED] {str(e)}")

                completed += 1
                # Live update after each attack
                pct = int((completed / total_tests) * 100)
                yield "\n".join(log_lines), f"{completed}/{total_tests} ({pct}%)", None

        # Process all detections in batch using Gemini API
        pending_count = get_pending_count()
        log_lines.append(f"\nRunning batch detection with Gemini API...")
        log_lines.append(f"\nPending detections: {pending_count}")
        yield "\n".join(log_lines), f"Processing {pending_count} detections...", None

        # Process all pending detections in batch
        process_pending_detections(batch_size=10)

        log_lines.append(f"\n[OK] Detection complete!")
        yield "\n".join(log_lines), f"Detection complete", None

        # Store results with detection outcomes
        log_lines.append("\nProcessing results...")
        yield "\n".join(log_lines), f"Processing results...", None

        for (
            detection_idx,
            model_config,
            attack,
            cat,
            inference_result,
        ) in pending_results:
            detection_result = get_pending_result(detection_idx)

            # Store result in collector
            collector.add_result(
                model_config=model_config,
                attack=attack,
                attack_category=cat.category,
                inference_result=inference_result,
                detection_result=detection_result,
            )

        log_lines.append(f"\n[OK] Processed {len(pending_results)} results")
        log_lines.append("\nExporting results...")

        # Generate summary
        model_metrics = collector.get_model_metrics()

        table_data = []
        for model_key, mm in model_metrics.items():
            # Add breakdown by filtering results for this model
            model_results = [
                r for r in collector.results if r.model_key == model_key]

            # Calculate ASR per cell for this model
            for vector in ["direct", "indirect"]:
                for objective in [
                    "instruction_override",
                    "data_extraction",
                    "role_confusion",
                ]:
                    cell_results = [
                        r
                        for r in model_results
                        if r.injection_vector == vector
                        and r.attack_objective == objective
                    ]

                    if cell_results:
                        total = len(cell_results)
                        successful = sum(
                            1 for r in cell_results if r.detection_result == "success"
                        )
                        blocked = total - successful
                        asr = (successful / total * 100) if total > 0 else 0.0

                        table_data.append(
                            {
                                "Model": mm.model_name,
                                "Vector": vector.title(),
                                "Objective": objective.replace("_", " ").title(),
                                "Tests": total,
                                "Success": successful,
                                "Blocked": blocked,
                                "ASR (%)": asr,
                            }
                        )

        comparison_df = pd.DataFrame(table_data)
        comparison_df["ASR (%)"] = comparison_df["ASR (%)"].round(1)

        # Export to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = collector.export_to_csv(f"attack_results_{timestamp}.csv")
        log_lines.append(f"\n[OK] Results exported to: `{csv_path}`")

        yield "\n".join(log_lines), f"Complete", comparison_df

    except Exception as e:
        logger.error(f"\nError in attack batch test: {e}")
        yield f"[ERROR] {str(e)}", "Error", None
    finally:
        app_state.set_batch_running(False)


def create_attack_testing_tab(model_choices, category_choices, app_state):
    """Create the Attack Testing tab UI component."""

    # Split categories into direct and indirect
    direct_categories = [
        (c[0], c[1]) for c in category_choices if c[1].startswith("direct_")
    ]
    indirect_categories = [
        (c[0], c[1]) for c in category_choices if c[1].startswith("indirect_")
    ]

    with gr.TabItem("Attack Testing", id="batch"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                ### Automated Attack Testing
                Run all selected attacks against all selected models automatically.
                """
                )

                gr.Markdown("#### Configuration")

                model_checkboxes = gr.CheckboxGroup(
                    choices=[m[0] for m in model_choices],
                    value=[m[0] for m in model_choices],
                    label="Models to Test",
                )

                gr.Markdown("#### Attack Categories")

                direct_checkboxes = gr.CheckboxGroup(
                    choices=[c[0] for c in direct_categories],
                    value=[c[0] for c in direct_categories],
                    label="Direct Injection",
                )

                indirect_checkboxes = gr.CheckboxGroup(
                    choices=[c[0] for c in indirect_categories],
                    value=[c[0] for c in indirect_categories],
                    label="Indirect Injection",
                )

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
                gr.Markdown("#### Result Summary")
                batch_summary = gr.Dataframe(
                    headers=[
                        "Model",
                        "Vector",
                        "Objective",
                        "Tests",
                        "Success",
                        "Blocked",
                        "ASR (%)",
                    ],
                    interactive=False,
                )

        # Convert display names back to keys for batch testing
        def batch_test_wrapper(
            models_display, direct_categories_display, indirect_categories_display
        ):
            # Convert display names to keys
            model_key_map = {m[0]: m[1] for m in model_choices}
            direct_key_map = {c[0]: c[1] for c in direct_categories}
            indirect_key_map = {c[0]: c[1] for c in indirect_categories}

            model_keys = [
                model_key_map[m] for m in models_display if m in model_key_map
            ]
            direct_keys = [
                direct_key_map[c]
                for c in direct_categories_display
                if c in direct_key_map
            ]
            indirect_keys = [
                indirect_key_map[c]
                for c in indirect_categories_display
                if c in indirect_key_map
            ]

            # Yield from the generator to pass through live updates
            yield from run_batch_test(model_keys, direct_keys, indirect_keys, app_state)

        batch_btn.click(
            fn=batch_test_wrapper,
            inputs=[model_checkboxes, direct_checkboxes, indirect_checkboxes],
            outputs=[batch_log, progress_text, batch_summary],
            show_progress="hidden",
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
            """,
        )
