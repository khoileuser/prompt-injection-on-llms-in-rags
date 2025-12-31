# =============================================================================
# Visualization Tab Component
# =============================================================================

import base64
import tempfile
import logging
from typing import Tuple, Optional
from pathlib import Path

import gradio as gr

from src.visualization import get_visualizer
from src.metrics import get_metrics_collector, MetricsCollector

logger = logging.getLogger(__name__)


def generate_visualizations() -> Tuple[str, str, str, str, str, str]:
    """
    Generate all visualizations from current metrics.
    
    If no data in current session, attempts to load from latest results file.
    
    Returns:
        Tuple of (dashboard, bar_chart, heatmap, line_chart, radar_chart, defense_comparison)
        All as file paths to temporary PNG files
    """
    visualizer = get_visualizer()
    collector = get_metrics_collector()
    
    # If no results in current session, try to load from latest file
    if not collector.results:
        try:
            latest_file = _get_latest_results_file()
            if latest_file:
                logger.info(f"Loading results from {latest_file}")
                collector.load_from_file(latest_file)
        except Exception as e:
            logger.warning(f"Could not load previous results: {e}")
    
    if not collector.results:
        empty_msg = "No data available. Run some tests first!"
        return empty_msg, None, None, None, None, None
    
    try:
        # Generate base64 images
        dashboard_b64 = visualizer.create_summary_dashboard()
        bar_chart_b64 = visualizer.create_asr_bar_chart()
        heatmap_b64 = visualizer.create_heatmap()
        line_chart_b64 = visualizer.create_trend_line_chart()
        radar_chart_b64 = visualizer.create_radar_chart()
        defense_comparison_b64 = visualizer.create_defense_comparison_chart()
        
        # Convert base64 to temporary files
        def save_base64_to_temp(b64_str: str, prefix: str) -> str:
            """Save base64 image to a temporary file and return the path."""
            img_data = base64.b64decode(b64_str)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'{prefix}_')
            temp_file.write(img_data)
            temp_file.close()
            return temp_file.name
        
        dashboard_path = save_base64_to_temp(dashboard_b64, 'dashboard')
        bar_chart_path = save_base64_to_temp(bar_chart_b64, 'bar_chart')
        heatmap_path = save_base64_to_temp(heatmap_b64, 'heatmap')
        line_chart_path = save_base64_to_temp(line_chart_b64, 'line_chart')
        radar_chart_path = save_base64_to_temp(radar_chart_b64, 'radar_chart')
        defense_comparison_path = save_base64_to_temp(defense_comparison_b64, 'defense_comparison')
        
        return (
            dashboard_path,
            bar_chart_path,
            heatmap_path,
            line_chart_path,
            radar_chart_path,
            defense_comparison_path
        )
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating visualizations: {str(e)}", None, None, None, None, None


def _get_latest_results_file() -> Optional[str]:
    """
    Find the most recent results JSON file in the results directory.
    
    Returns:
        Path to the latest results file, or None if not found
    """
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # Find all JSON files
    json_files = list(results_dir.glob("results_*.json"))
    if not json_files:
        return None
    
    # Sort by modification time (most recent first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(json_files[0])


def get_metrics_summary() -> str:
    """Get current metrics summary as markdown."""
    collector = get_metrics_collector()
    
    if not collector.results:
        # Try loading latest results
        try:
            latest_file = _get_latest_results_file()
            if latest_file:
                collector.load_from_file(latest_file)
                if collector.results:
                    return _format_metrics_summary(collector, from_file=latest_file)
        except Exception as e:
            logger.warning(f"Could not load results: {e}")
        
        return "No test results available. Run some tests first!"
    
    return _format_metrics_summary(collector, from_file=None)


def _format_metrics_summary(collector: MetricsCollector, from_file: Optional[str] = None) -> str:
    """Format metrics summary as markdown."""
    summary = collector.get_summary()
    
    data_source = ""
    if from_file:
        from pathlib import Path
        filename = Path(from_file).name
        data_source = f"\n**Data Source:** {filename} (auto-loaded)"
    
    return f"""
## Test Statistics
- **Total Tests Run:** {summary.get('total_tests', 0)}
- **Models Tested:** {summary.get('unique_models', 0)}
- **Attack Categories:** {summary.get('unique_attack_categories', 0)}

## Key Findings
- **Overall ASR:** {summary.get('overall_asr', 0)*100:.1f}%
- **Most Robust Model:** {summary.get('most_robust_model', 'N/A')}
- **Most Vulnerable Model:** {summary.get('most_vulnerable_model', 'N/A')}
- **Most Effective Attack:** {summary.get('most_effective_attack', 'N/A')}
- **Least Effective Attack:** {summary.get('least_effective_attack', 'N/A')}
"""


def create_visualization_tab():
    """Create the Visualization tab UI component."""
    
    with gr.TabItem("Visualization", id="viz"):
        gr.Markdown("""
        ### Results Visualization
        Generate charts and visualizations from test results.
        """)
        
        with gr.Row():
            refresh_btn = gr.Button("Refresh Visualizations", variant="primary")
        
        with gr.Row():
            metrics_summary = gr.Markdown(get_metrics_summary())
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Summary Dashboard")
                dashboard_img = gr.Image(label="Complete Dashboard Overview", type="filepath")
            with gr.Column():
                gr.Markdown("### Defense Analysis")
                defense_comparison_img = gr.Image(label="Defense Strategy Comparison", type="filepath")
        
        gr.Markdown("---")
        gr.Markdown("### Model Performance")
        
        with gr.Row():
            with gr.Column():
                bar_chart_img = gr.Image(label="ASR by Model", type="filepath")
            with gr.Column():
                radar_chart_img = gr.Image(label="Robustness Radar", type="filepath")
        
        gr.Markdown("---")
        gr.Markdown("### Attack Analysis")
        
        with gr.Row():
            with gr.Column():
                heatmap_img = gr.Image(label="Model × Attack Heatmap", type="filepath")
            with gr.Column():
                line_chart_img = gr.Image(label="ASR Trend by Variation", type="filepath")
        
        def refresh_viz():
            summary = get_metrics_summary()
            viz = generate_visualizations()
            return (summary,) + viz
        
        refresh_btn.click(
            fn=refresh_viz,
            outputs=[metrics_summary, dashboard_img, bar_chart_img, heatmap_img, 
                    line_chart_img, radar_chart_img, defense_comparison_img]
        )
