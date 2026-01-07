from src.ui.settings_tab import create_settings_tab
from src.ui.visualization_tab import create_visualization_tab
from src.ui.batch_testing_tab import create_batch_testing_tab
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


def get_model_choices() -> List[Tuple[str, str]]:
    """Get list of (display_name, key) tuples for model dropdown."""
    models = get_models()
    return [(m.name, m.key) for m in models]


def get_attack_choices() -> List[Tuple[str, str]]:
    """Get list of (display_name, id) tuples for attack dropdown."""
    attacks = get_attacks()
    return [(f"[{a.id}] {a.name}", a.id) for a in attacks]


def get_category_choices() -> List[Tuple[str, str]]:
    """Get list of attack category choices organized by injection vector and objective."""
    categories = get_attack_categories()

    # Organize by injection vector and objective
    choices = []

    # Direct injection categories
    direct_categories = [c for c in categories if c.key.startswith('direct_')]
    if direct_categories:
        for cat in sorted(direct_categories, key=lambda x: x.key):
            # Extract objective from key (e.g., 'direct_instruction_override' -> 'Instruction Override')
            objective = cat.key.replace(
                'direct_', '').replace('_', ' ').title()
            choices.append((objective, cat.key))

    # Indirect injection categories
    indirect_categories = [
        c for c in categories if c.key.startswith('indirect_')]
    if indirect_categories:
        for cat in sorted(indirect_categories, key=lambda x: x.key):
            # Extract objective from key (e.g., 'indirect_data_extraction' -> 'Data Extraction')
            objective = cat.key.replace(
                'indirect_', '').replace('_', ' ').title()
            choices.append((objective, cat.key))

    return choices


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
    footer{display:none !important}
    
    #batch-log-wrapper {
        max-height: 400px;
        overflow-y: auto !important;
        scroll-behavior: smooth;
    }
    
    #batch-log-wrapper > div {
        overflow: visible !important;
    }
    
    /* Make code blocks wrap text instead of horizontal scroll */
    #batch-log-wrapper pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-x: auto !important;
    }
    #batch-log-wrapper code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    
    /* Disable scrollbar for batch log */
    #batch-log-wrapper::-webkit-scrollbar {
        width: 0px;
        height: 0px;
    }

    #batch-log-wrapper::-webkit-scrollbar-thumb {
        border-radius: 0px;
    }
    """

    with gr.Blocks(title="Prompt Injection Security Research") as app:
        with gr.Tabs():
            # Create all tabs
            create_live_demo_tab(model_choices, attack_choices)
            create_batch_testing_tab(
                model_choices, category_choices, app_state)
            create_visualization_tab()
            create_settings_tab()

    return app, custom_css
