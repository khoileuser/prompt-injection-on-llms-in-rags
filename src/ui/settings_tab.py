# =============================================================================
# Settings Tab Component
# =============================================================================

import gradio as gr

from src.config_loader import get_models, get_attacks, get_attack_categories
from src.metrics import get_metrics_collector
from src.detection import (
    get_detector,
    set_detection_method,
    DETECTION_METHODS,
)


def create_settings_tab():
    """Create the Settings tab UI component."""

    with gr.TabItem("Settings", id="settings"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### System Information")

                import torch
                cuda_available = torch.cuda.is_available()
                cuda_info = ""
                if cuda_available:
                    cuda_info = f"""
- **GPU:** {torch.cuda.get_device_name(0)}
- **VRAM:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB
"""
                else:
                    cuda_info = "- **GPU:** Not available (using CPU)"

                gr.Markdown(f"""
**Device Configuration:**
{cuda_info}

**Loaded Models:** {len(get_models())} configured

**Attack Vectors:** {len(get_attacks())} total across {len(get_attack_categories())} categories
""")

            with gr.Column():
                gr.Markdown("#### Quick Actions")

                def clear_results():
                    get_metrics_collector().clear()
                    return "[OK] Results cleared successfully"

                clear_btn = gr.Button("Clear All Results")
                clear_status = gr.Markdown("")

                clear_btn.click(fn=clear_results, outputs=[clear_status])

        # Detection Method Settings
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
#### Detection Method Settings

Choose the detection method for identifying successful prompt injection attacks:

- **llama_guard**: Uses Meta's LLaMA Guard 3 model for accurate safety classification.
- **regex**: Fast, traditional pattern-based detection. May have more false positives.
""")

                detection_method_dropdown = gr.Dropdown(
                    choices=list(DETECTION_METHODS.keys()),
                    value="llama_guard",
                    label="Detection Method",
                    info="Select the detection method to use for attack classification"
                )

                def update_detection_method(method):
                    """Update the detection method."""
                    set_detection_method(method)

                detection_method_dropdown.change(
                    fn=update_detection_method,
                    inputs=[detection_method_dropdown],
                )

