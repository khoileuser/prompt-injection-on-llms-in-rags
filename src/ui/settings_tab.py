# =============================================================================
# Settings Tab Component
# =============================================================================

import gradio as gr

from src.config_loader import get_models, get_attacks, get_attack_categories
from src.metrics import get_metrics_collector


def create_settings_tab():
    """Create the Settings tab UI component."""
    
    with gr.TabItem("Settings", id="settings"):
        gr.Markdown("""
        ### Application Settings
        
        Configure the application behavior and view system information.
        """)
        
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
