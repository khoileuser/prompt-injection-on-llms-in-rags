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
from src.defense_prompts import (
    DefenseStrategy,
    set_defense_strategy,
    get_defense_builder,
)

logger = logging.getLogger(__name__)


def run_comparative_batch_test(
    selected_models: List[str],
    selected_direct_categories: List[str],
    selected_indirect_categories: List[str],
    selected_defense_strategies: List[str],
    app_state,
):
    """
    Run batch tests across selected defense strategies for comparative analysis.

    Uses batch detection at the end to minimize Gemini API calls.

    Args:
        selected_models: List of model keys to test
        selected_direct_categories: List of direct attack category keys to test
        selected_indirect_categories: List of indirect attack category keys to test
        selected_defense_strategies: List of defense strategy values to test
        app_state: Application state object

    Yields:
        Tuple of (progress_text, status_log, comparison_df) with live updates
    """
    if not selected_models:
        yield "Ready", "[WARNING] Please select at least one model", None
        return

    # Combine selected categories
    selected_categories = selected_direct_categories + selected_indirect_categories

    if not selected_categories:
        yield "Ready", "[WARNING] Please select at least one attack category", None
        return

    if not selected_defense_strategies:
        yield "Ready", "[WARNING] Please select at least one defense strategy", None
        return

    if app_state.is_batch_running:
        yield "Busy", "[WARNING] A batch test is already running", None
        return

    app_state.set_batch_running(True)

    # Clear any pending detections from previous runs
    clear_pending_detections()

    try:
        # Get attacks for selected categories
        all_categories = get_attack_categories()
        attacks_to_run = []

        for cat in all_categories:
            if cat.key in selected_categories:
                for var in cat.variations:
                    attacks_to_run.append((cat, var))

        # Filter to only selected defense strategies
        all_strategies = [
            s for s in DefenseStrategy if s.value in selected_defense_strategies]
        total_tests = len(selected_models) * \
            len(attacks_to_run) * len(all_strategies)
        completed = 0

        # Storage for pending results (detection_idx -> metadata for processing later)
        pending_results = []

        log_lines = [
            "Batch Testing",
            f"- Models: {len(selected_models)}",
            f"- Attack Objectives: {len(selected_categories)}",
            f"- Total Tests: {total_tests}",
        ]

        # Initial yield
        yield f"Starting... 0/{total_tests}", "\n".join(log_lines), None

        # Run all attacks and collect responses
        log_lines.append("\nRunning attacks across all defense strategies...")
        yield f"Running attacks...", "\n".join(log_lines), None

        # ITERATE THROUGH EACH DEFENSE MECHANISM
        for strategy_idx, defense_strategy in enumerate(all_strategies, 1):
            set_defense_strategy(defense_strategy)
            defense_config = get_defense_builder().get_config()

            log_lines.append("\n")
            log_lines.append(
                f"\nMechanism {strategy_idx}/{len(all_strategies)}: {defense_config.name}")
            log_lines.append(f"\n*{defense_config.description}*")
            yield f"Mechanism {strategy_idx}/{len(all_strategies)}: {defense_strategy.value}", "\n".join(log_lines), None

            # Run tests for each model
            for model_key in selected_models:
                log_lines.append(f"\nLoading model: {model_key}")
                yield f"Loading model... {completed}/{total_tests}", "\n".join(log_lines), None

                success, msg = load_model(model_key)
                if not success:
                    log_lines.append(f"\n[FAILED] Failed to load model: {msg}")
                    yield f"Model failed {completed}/{total_tests}", "\n".join(log_lines), None
                    continue

                model_config = get_config_loader().get_model(model_key)
                log_lines.append(f"\n[OK] Model loaded: {model_config.name}")
                yield f"Model loaded {completed}/{total_tests}", "\n".join(log_lines), None

                # Run attacks
                for cat, attack in attacks_to_run:
                    # Show attack info
                    log_lines.append("\n")
                    log_lines.append(
                        f"\nTest {completed + 1}/{total_tests}: {attack.name}")
                    log_lines.append(f"\nObjective: {cat.category}")

                    # Show prompt being sent
                    prompt_preview = attack.prompt
                    prompt_preview = prompt_preview.replace("```", "\\`\\`\\`")
                    log_lines.append(f"\nPrompt:\n```\n{prompt_preview}\n```")
                    yield f"Test {completed + 1}/{total_tests}", "\n".join(log_lines), None

                    try:
                        # Generate response with current defense strategy and attack object
                        inference_result = generate_response(
                            attack.prompt, attack=attack
                        )

                        if inference_result and inference_result.success:
                            # Add to pending detection queue (will be processed in batch later)
                            detection_idx = add_pending_detection(
                                attack, inference_result.response, cat.category
                            )

                            # Store metadata for later processing
                            pending_results.append(
                                {
                                    "detection_idx": detection_idx,
                                    "defense_strategy": defense_strategy,
                                    "defense_config": defense_config,
                                    "model_config": model_config,
                                    "model_key": model_key,
                                    "attack": attack,
                                    "cat": cat,
                                    "inference_result": inference_result,
                                }
                            )

                            # Show response from LLM
                            response_preview = inference_result.response
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
                    yield f"{completed}/{total_tests} ({pct}%)", "\n".join(log_lines), None

        # Process all detections in batch
        pending_count = get_pending_count()
        log_lines.append(f"\nRunning batch detection with Gemini API...")
        log_lines.append(f"\nPending detections: {pending_count}")
        yield f"Processing {pending_count} detections...", "\n".join(log_lines), None

        # Process all pending detections in batch
        process_pending_detections(batch_size=10)

        log_lines.append(f"\n[OK] Detection complete!")
        yield f"Detection complete", "\n".join(log_lines), None

        # Process results and generate statistics
        log_lines.append("\nProcessing results...")
        yield f"Processing results...", "\n".join(log_lines), None

        # Storage for comparative results
        comparative_results = []

        # Track statistics by strategy
        strategy_stats = {}  # strategy_value -> {success: int, total: int}

        collector = get_metrics_collector()

        for item in pending_results:
            detection_result = get_pending_result(item["detection_idx"])

            defense_strategy = item["defense_strategy"]
            defense_config = item["defense_config"]
            model_config = item["model_config"]
            attack = item["attack"]
            cat = item["cat"]
            inference_result = item["inference_result"]

            # Add to metrics collector
            collector.add_result(
                model_config=model_config,
                attack=attack,
                attack_category=cat.category,
                inference_result=inference_result,
                detection_result=detection_result,
                defense_strategy=defense_strategy.value,
            )

            # Track statistics
            is_success = detection_result.result.value == "success"

            if defense_strategy.value not in strategy_stats:
                strategy_stats[defense_strategy.value] = {
                    "success": 0,
                    "total": 0,
                    "name": defense_config.name,
                }
            strategy_stats[defense_strategy.value]["total"] += 1
            if is_success:
                strategy_stats[defense_strategy.value]["success"] += 1

            # Store for comparative analysis
            comparative_results.append(
                {
                    "defense_strategy": defense_config.name,
                    "defense_value": defense_strategy.value,
                    "model": model_config.name,
                    "model_key": item["model_key"],
                    "attack_id": attack.id,
                    "attack_name": attack.name,
                    "attack_category": cat.category,
                    "result": detection_result.result.value,
                    "success": is_success,
                    "response_time": inference_result.response_time,
                }
            )

        log_lines.append(f"\n[OK] Processed {len(pending_results)} results")
        log_lines.append("\nExporting results...")

        # Generate summary
        df = pd.DataFrame(comparative_results)

        # Define custom ordering for defense strategies
        defense_order = {
            "Strong System-Prompt Prefixing": 1,
            "Source Tagging/Quoting": 2,
            "Output Filtering": 3,
        }

        # Calculate ASR per defense strategy
        asr_by_defense = df.groupby("defense_strategy")["success"].mean() * 100
        # Sort by custom order instead of ASR
        asr_by_defense = asr_by_defense.reindex(
            sorted(asr_by_defense.index, key=lambda x: defense_order.get(x, 999))
        )

        summary_data = []
        for defense in asr_by_defense.index:
            defense_df = df[df["defense_strategy"] == defense]

            # Create rows for each model within this defense strategy
            for model in defense_df["model"].unique():
                model_df = defense_df[defense_df["model"] == model]

                # Group by attack category
                for category in model_df["attack_category"].unique():
                    cat_df = model_df[model_df["attack_category"] == category]

                    if len(cat_df) > 0:
                        cat_total = len(cat_df)
                        cat_success = int(cat_df["success"].sum())
                        cat_blocked = int(
                            (cat_df["result"] == "blocked").sum())
                        cat_asr = (
                            (cat_success / cat_total * 100) if cat_total > 0 else 0
                        )

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

                        summary_data.append(
                            {
                                "Defense": defense,
                                "Model": model,
                                "Vector": vector,
                                "Objective": objective,
                                "Tests": cat_total,
                                "Success": cat_success,
                                "Blocked": cat_blocked,
                                "ASR (%)": cat_asr,
                            }
                        )

        comparison_table = pd.DataFrame(summary_data)
        if not comparison_table.empty:
            comparison_table["ASR (%)"] = comparison_table["ASR (%)"].round(1)

            # Apply custom defense order to the table
            comparison_table["defense_order"] = comparison_table["Defense"].map(
                defense_order
            )
            comparison_table = comparison_table.sort_values(
                ["defense_order", "Model", "Vector", "Objective"]
            )
            comparison_table = comparison_table.drop("defense_order", axis=1)

        # Export to CSV
        collector = get_metrics_collector()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = collector.export_to_csv(f"results_{timestamp}.csv")
        log_lines.append(f"\n[OK] Results exported: `{csv_path}`")

        yield f"Complete", "\n".join(log_lines), comparison_table

    except Exception as e:
        logger.error(f"\nError in batch test: {e}")
        yield "Error", f"[ERROR] {str(e)}", None
    finally:
        app_state.set_batch_running(False)


def create_batch_testing_tab(model_choices, category_choices, app_state):
    """Create the Batch Testing tab UI component."""

    # Split categories into direct and indirect
    direct_categories = [
        (c[0], c[1]) for c in category_choices if c[1].startswith("direct_")
    ]
    indirect_categories = [
        (c[0], c[1]) for c in category_choices if c[1].startswith("indirect_")
    ]

    with gr.TabItem("Batch Testing", id="batch"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                ### Automated Batch Testing
                Automatically tests all defense mechanisms against selected attacks.
                """
                )

                gr.Markdown("#### Configuration")

                model_checkboxes = gr.CheckboxGroup(
                    choices=[m[0] for m in model_choices],
                    value=[m[0] for m in model_choices],
                    label="Models to Test",
                )

                # Defense strategies selection
                defense_strategy_choices = [
                    ("No Defense (Baseline)", DefenseStrategy.NONE.value),
                    ("Strong System-Prompt Prefixing",
                     DefenseStrategy.STRONG_PREFIX.value),
                    ("Source Tagging/Quoting", DefenseStrategy.SOURCE_TAGGING.value),
                    ("Output Filtering", DefenseStrategy.OUTPUT_FILTERING.value),
                ]

                defense_checkboxes = gr.CheckboxGroup(
                    choices=[d[0] for d in defense_strategy_choices],
                    value=[d[0] for d in defense_strategy_choices],
                    label="Defense Strategies",
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

                comparative_btn = gr.Button(
                    "Start Batch Testing", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                gr.Markdown("#### Progress")
                progress_text = gr.Markdown("", elem_id="progress-text")

                with gr.Accordion("Test Log", open=True):
                    with gr.Column(elem_id="batch-log-wrapper"):
                        batch_log = gr.Markdown(value="Ready to start...")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Result Summary")
                comparison_table_ui = gr.Dataframe(
                    headers=[
                        "Defense",
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

        # Convert display names back to keys
        def comparative_test_wrapper(
            models_display, defense_display, direct_categories_display, indirect_categories_display
        ):
            model_key_map = {m[0]: m[1] for m in model_choices}
            defense_value_map = {d[0]: d[1] for d in defense_strategy_choices}
            direct_key_map = {c[0]: c[1] for c in direct_categories}
            indirect_key_map = {c[0]: c[1] for c in indirect_categories}

            model_keys = [
                model_key_map[m] for m in models_display if m in model_key_map
            ]
            defense_values = [
                defense_value_map[d] for d in defense_display if d in defense_value_map
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

            yield from run_comparative_batch_test(
                model_keys, direct_keys, indirect_keys, defense_values, app_state
            )

        comparative_btn.click(
            fn=comparative_test_wrapper,
            inputs=[model_checkboxes, defense_checkboxes,
                    direct_checkboxes, indirect_checkboxes],
            outputs=[progress_text, batch_log, comparison_table_ui],
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
