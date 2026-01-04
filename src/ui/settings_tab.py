import os
import csv
from pathlib import Path

import gradio as gr

from src.config_loader import get_models, get_attacks, get_attack_categories
from src.metrics import get_metrics_collector
from src.detection import get_detector, DetectionResult


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
                # Check if Gemini is configured
                detector = get_detector()
                gemini_status = "Configured" if detector.is_configured(
                ) else "Not configured (set GEMINI_API_KEY)"

                gr.Markdown(f"""
**Detection:** Gemini API ({os.getenv('GEMINI_MODEL', 'gemini-flash-latest')})

**API Key Status:** {gemini_status}
""")

        # Error Detection Retry Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Retry Error Detections")
                gr.Markdown("""
Select a CSV file from the results folder to retry error detections. 
Only rows with `detection_result = "error"` will be reprocessed.
The original file will be updated with corrected detections.
""")

                def get_csv_files():
                    """Get list of CSV files in results folder."""
                    results_dir = Path("results")
                    if not results_dir.exists():
                        return []
                    csv_files = sorted(
                        [f.name for f in results_dir.glob("*.csv")],
                        reverse=True  # Most recent first
                    )
                    return csv_files if csv_files else ["No CSV files found"]

                with gr.Row():
                    csv_file_dropdown = gr.Dropdown(
                        choices=get_csv_files(),
                        label="Select Results CSV",
                        info="Choose a file from the results folder",
                        interactive=True,
                        scale=2
                    )

                    refresh_btn = gr.Button("Reload")
                    refresh_btn.click(
                        fn=lambda: gr.Dropdown(choices=get_csv_files()),
                        outputs=[csv_file_dropdown]
                    )

                with gr.Row():
                    batch_size_input = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=8,
                        step=1,
                        label="Batch Size (lower = fewer rate limits)",
                        info="Number of detections per API call"
                    )
                    max_retries_input = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Max Retries per Batch",
                        info="Higher = more persistent but slower"
                    )

                retry_btn = gr.Button(
                    "Retry Error Detections", variant="primary")
                retry_progress = gr.Markdown("")

                def retry_error_detections(csv_filename, batch_size, max_retries):
                    """Retry only the error detections from a CSV file."""
                    if not csv_filename or csv_filename == "No CSV files found":
                        return "Please select a CSV file first."

                    detector = get_detector()
                    if not detector.is_configured():
                        return "Gemini API not configured. Please set GEMINI_API_KEY."

                    try:
                        # Construct full path
                        csv_path = Path("results") / csv_filename

                        if not csv_path.exists():
                            return f"File not found: {csv_path}"

                        # Read CSV and find error rows
                        error_rows = []
                        all_rows = []

                        with open(csv_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                all_rows.append(row)
                                if row.get('detection_result', '').lower() == 'error':
                                    error_rows.append(row)

                        if not error_rows:
                            return f"No error detections found in the CSV. Total rows: {len(all_rows)}"

                        yield f"Found {len(error_rows)} error detections out of {len(all_rows)} total rows. Starting retry..."

                        # Create a simple attack object for detection
                        class SimpleAttack:
                            def __init__(self, attack_id, name):
                                self.id = attack_id
                                self.name = name

                        # Add all errors to pending queue
                        detection_ids = []
                        for row in error_rows:
                            attack = SimpleAttack(
                                row.get('attack_id', 'unknown'),
                                row.get('attack_name', 'Unknown Attack')
                            )
                            det_id = detector.add_pending(
                                attack=attack,
                                response=row.get('response', ''),
                                category=row.get('attack_category', 'unknown')
                            )
                            detection_ids.append((det_id, row))

                        yield f"Processing {len(error_rows)} detections in batches of {batch_size}..."

                        # Process batch with configured settings
                        detector.process_batch(
                            batch_size=int(batch_size),
                            max_retries=int(max_retries)
                        )

                        yield f"Batch processing complete. Updating results..."

                        # Update the rows with new detection results
                        corrected_count = 0
                        still_error_count = 0

                        for det_id, original_row in detection_ids:
                            result = detector.get_result(det_id)
                            if result:
                                original_row['detection_result'] = result.result.value
                                original_row['explanation'] = result.reasoning

                                if result.result != DetectionResult.ERROR:
                                    corrected_count += 1
                                else:
                                    still_error_count += 1

                        # Overwrite the original CSV file
                        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                            if all_rows:
                                writer = csv.DictWriter(
                                    f, fieldnames=all_rows[0].keys())
                                writer.writeheader()
                                writer.writerows(all_rows)

                        summary = f"""
**Retry Complete**

- **File Updated:** `{csv_filename}`
- **Total Rows:** {len(all_rows)}
- **Error Rows Found:** {len(error_rows)}
- **Successfully Corrected:** {corrected_count}
- **Still Errors:** {still_error_count}

The original file has been updated with corrected detections.
"""
                        yield summary

                    except Exception as e:
                        yield f"Error during retry: {str(e)}"

                retry_btn.click(
                    fn=retry_error_detections,
                    inputs=[csv_file_dropdown, batch_size_input,
                            max_retries_input],
                    outputs=[retry_progress]
                )
