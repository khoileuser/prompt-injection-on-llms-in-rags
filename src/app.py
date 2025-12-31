# =============================================================================
# Gradio Web Application for Prompt Injection Security Research
# =============================================================================
# This module provides a web-based interface for:
# 1. Live Demo: Interactive testing of individual attacks against models
# 2. Attack Testing: Automated testing of all attacks across all models
# 3. Defense Testing: Automated testing across ALL defense strategies
# 4. Visualization: Interactive charts and dashboards
# 5. Export: CSV and JSON export of results
#
# The interface is designed for security researchers to easily conduct
# and document prompt injection vulnerability assessments.
# =============================================================================

from src.ui.settings_tab import create_settings_tab
from src.ui.visualization_tab import create_visualization_tab
from src.ui.defense_testing_tab import create_defense_testing_tab
from src.ui.attack_testing_tab import create_attack_testing_tab
from src.ui.live_demo_tab import create_live_demo_tab
from src.config_loader import get_models, get_attacks, get_attack_categories
import os
import sys
import logging
import threading
from typing import Optional, List, Tuple

import gradio as gr

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Global State
# =============================================================================

class AppState:
    """Global application state manager."""

    def __init__(self):
        self.current_model: Optional[str] = None
        self.is_batch_running: bool = False
        self.batch_progress: float = 0.0
        self.batch_status: str = ""
        self._lock = threading.Lock()

    def set_batch_running(self, running: bool):
        with self._lock:
            self.is_batch_running = running

    def update_batch_progress(self, progress: float, status: str):
        with self._lock:
            self.batch_progress = progress
            self.batch_status = status


app_state = AppState()


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_choices() -> List[Tuple[str, str]]:
    """Get list of (display_name, key) tuples for model dropdown."""
    models = get_models()
    return [(m.name, m.key) for m in models]


def get_attack_choices() -> List[Tuple[str, str]]:
    """Get list of (display_name, id) tuples for attack dropdown."""
    attacks = get_attacks()
    return [(f"[{a.id}] {a.name}", a.id) for a in attacks]


def get_category_choices() -> List[Tuple[str, str]]:
    """Get list of attack category choices."""
    categories = get_attack_categories()
    return [(c.category, c.key) for c in categories]


# =============================================================================
# Build Gradio Interface
# =============================================================================

def create_app() -> gr.Blocks:
    """
    Create and configure the Gradio web application.

    Returns:
        Configured Gradio Blocks application
    """

    # Get initial choices
    model_choices = get_model_choices()
    attack_choices = get_attack_choices()
    category_choices = get_category_choices()

    # Custom CSS for styling
    custom_css = """
    .warning-box { background-color: #fef3c7; border: 1px solid #f59e0b; padding: 10px; border-radius: 5px; }
    .success-box { background-color: #d1fae5; border: 1px solid #10b981; padding: 10px; border-radius: 5px; }
    .error-box { background-color: #fee2e2; border: 1px solid #ef4444; padding: 10px; border-radius: 5px; }
    #batch-log-wrapper { 
        max-height: 400px; 
        overflow-y: auto !important;
        scroll-behavior: smooth;
    }
    #batch-log-wrapper > div {
        overflow: visible !important;
    }
    """

    with gr.Blocks(title="Prompt Injection Security Research") as app:
        with gr.Tabs():
            # Create all tabs
            create_live_demo_tab(model_choices, attack_choices)
            create_attack_testing_tab(
                model_choices, category_choices, app_state)
            create_defense_testing_tab(
                model_choices, category_choices, app_state)
            create_visualization_tab()
            create_settings_tab()

    return app, custom_css

# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Launch the Gradio web application."""
    app, custom_css = create_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=custom_css
    )


if __name__ == "__main__":
    main()
