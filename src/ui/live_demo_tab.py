import time
import logging
from typing import Tuple

import gradio as gr

from src.config_loader import (
    get_config_loader, get_attack_categories, AttackVariation
)
from src.inference import get_model_manager, load_model, generate_response
from src.detection import detect_attack
from src.metrics import get_metrics_collector
from src.defense_prompts import (
    get_defense_builder, set_defense_strategy, get_defense_strategy,
    get_all_defense_configs, get_defense_description, DefenseStrategy
)

logger = logging.getLogger(__name__)


def format_detection_result(result) -> str:
    """Format detection result for display."""
    labels = {
        "success": "SUCCESS (Attack worked)",
        "blocked": "BLOCKED (Attack failed)",
        "error": "ERROR (Detection failed)"
    }
    return labels.get(result.result.value, f"Unknown: {result.result}")


def load_model_handler(model_key: str) -> Tuple[str, str]:
    """
    Handle model loading from the UI.

    Args:
        model_key: The model key to load

    Returns:
        Tuple of (status_message, model_info)
    """
    if not model_key:
        return "[WARNING] Please select a model", ""

    logger.info(f"Loading model: {model_key}")

    success, message = load_model(model_key)

    if success:
        model_config = get_config_loader().get_model(model_key)
        model_info = f"""
**Model Loaded Successfully**

- **Name:** {model_config.name}
- **HuggingFace ID:** `{model_config.model_id}`
- **Parameters:** {model_config.parameters}B
- **Description:** {model_config.description}

Ready to test attacks!
"""
        return f"[OK] {message}", model_info
    else:
        return f"[FAILED] {message}", ""


def get_attack_info(attack_id: str) -> str:
    """Get detailed information about an attack."""
    if not attack_id:
        return ""

    attack = get_config_loader().get_attack_by_id(attack_id)
    if not attack:
        return "Attack not found"

    # Find category
    for category in get_attack_categories():
        for var in category.variations:
            if var.id == attack_id:
                return f"""
**Attack Information**

- **ID:** {attack.id}
- **Name:** {attack.name}
- **Category:** {category.category}

**Description:**
{attack.description}
"""
    return ""


def update_defense_strategy(strategy_value: str) -> str:
    """Update the defense strategy and return description."""
    try:
        strategy = DefenseStrategy(strategy_value)
        set_defense_strategy(strategy)
        return get_defense_description(strategy_value)
    except ValueError:
        return "Invalid defense strategy"


def run_single_attack(
    model_key: str,
    attack_id: str,
    custom_prompt: str = "",
    defense_strategy: str = "none"
) -> Tuple[str, str, str, str]:
    """
    Run a single attack and return results.

    Args:
        model_key: The model to test
        attack_id: The attack to run
        custom_prompt: Optional custom prompt (overrides attack prompt)
        defense_strategy: The defense strategy to apply

    Returns:
        Tuple of (status, prompt_used, response, detection_result)
    """
    # Update defense strategy
    try:
        strategy = DefenseStrategy(defense_strategy)
        set_defense_strategy(strategy)
    except ValueError:
        set_defense_strategy(DefenseStrategy.NONE)

    # Validate inputs
    if not model_key:
        return "[WARNING] Please select a model first", "", "", ""

    if not attack_id and not custom_prompt:
        return "[WARNING] Please select an attack or enter a custom prompt", "", "", ""

    # Auto-load model if not already loaded
    model_manager = get_model_manager()
    if model_manager.current_model_key != model_key:
        logger.info(f"Auto-loading model: {model_key}")
        success, message = load_model(model_key)
        if not success:
            return f"[FAILED] Could not load model: {message}", "", "", ""

    # Get attack and model config
    model_config = get_config_loader().get_model(model_key)

    if attack_id:
        attack = get_config_loader().get_attack_by_id(attack_id)
        if not attack:
            return "[WARNING] Attack not found", "", "", ""
        prompt = custom_prompt if custom_prompt else attack.prompt
    else:
        # Custom prompt only
        prompt = custom_prompt
        attack = AttackVariation(
            id="custom",
            name="Custom Prompt",
            prompt=prompt,
            description="User-provided custom prompt"
        )

    # Get current defense strategy for logging
    current_defense = get_defense_strategy()
    defense_config = get_defense_builder().get_config()
    logger.info(
        f"Running attack {attack.id} against {model_key} with defense: {current_defense.value}")
    start_time = time.time()

    try:
        # Pass attack object for document injection support
        inference_result = generate_response(prompt, attack=attack)

        if not inference_result.success:
            return (
                f"[FAILED] Inference error: {inference_result.error_message}",
                prompt, "", ""
            )

        # Run detection
        category_name = ""
        for cat in get_attack_categories():
            for var in cat.variations:
                if var.id == attack.id:
                    category_name = cat.category
                    break

        detection_result = detect_attack(
            attack, inference_result.response, category_name)

        # Add to metrics
        collector = get_metrics_collector()
        collector.add_result(
            model_config=model_config,
            attack=attack,
            attack_category=category_name,
            inference_result=inference_result,
            detection_result=detection_result
        )

        # Format results with defense info
        status = f"""
**Test Complete**

- **Defense Strategy:** {defense_config.name}
- Response Time: {inference_result.response_time:.2f}s
- Tokens Generated: {inference_result.tokens_generated}

**Result:** {format_detection_result(detection_result)}

{detection_result.reasoning}
"""

        detection_display = f"""
**Detection Result:** {format_detection_result(detection_result)}

**Defense Used:** {defense_config.name}

**Explanation:** {detection_result.reasoning}
"""

        return status, prompt, inference_result.response, detection_display

    except Exception as e:
        logger.error(f"Error running attack: {e}")
        return f"[FAILED] Error: {str(e)}", prompt, "", ""


def create_live_demo_tab(model_choices, attack_choices):
    """Create the Live Demo tab UI component."""

    with gr.TabItem("Live Demo", id="demo"):
        gr.Markdown("""
        ### Interactive Attack Testing
        Test individual prompt injection attacks against loaded models in real-time.
        Compare results with different **defense strategies** to evaluate their effectiveness.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Model Selection
                gr.Markdown("#### 1. Select Model")
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="Model",
                    info="Model will load automatically when you run an attack"
                )

                # Defense Strategy Selection
                gr.Markdown("#### 2. Select Defense Strategy")
                defense_dropdown = gr.Dropdown(
                    choices=get_all_defense_configs(),
                    value="none",
                    label="Defense Strategy",
                    info="Select a defense prompt strategy (None = baseline)"
                )
                defense_info = gr.Markdown(get_defense_description("none"))

                # Attack Selection
                gr.Markdown("#### 3. Select Attack")
                attack_dropdown = gr.Dropdown(
                    choices=attack_choices,
                    label="Attack",
                    info="Select a predefined attack or use custom prompt"
                )
                attack_info = gr.Markdown("")

                custom_prompt = gr.Textbox(
                    label="Custom Prompt (Optional)",
                    placeholder="Enter a custom prompt to test...",
                    lines=3
                )

                run_btn = gr.Button("Run Attack", variant="primary")

            with gr.Column(scale=1):
                # Results
                gr.Markdown("#### Results")
                result_status = gr.Markdown("")

                prompt_display = gr.Textbox(
                    label="Prompt Sent",
                    interactive=False,
                    lines=5
                )

                response_display = gr.Textbox(
                    label="Model Response",
                    interactive=False,
                    lines=10
                )

                with gr.Accordion("Detection Analysis", open=True):
                    detection_display = gr.Markdown("")

        # Event handlers
        defense_dropdown.change(
            fn=update_defense_strategy,
            inputs=[defense_dropdown],
            outputs=[defense_info]
        )

        attack_dropdown.change(
            fn=get_attack_info,
            inputs=[attack_dropdown],
            outputs=[attack_info]
        )

        run_btn.click(
            fn=run_single_attack,
            inputs=[model_dropdown, attack_dropdown,
                    custom_prompt, defense_dropdown],
            outputs=[result_status, prompt_display,
                     response_display, detection_display]
        )
