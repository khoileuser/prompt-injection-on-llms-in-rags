# =============================================================================
# Prompt Injection Security Research
# Main Entry Point
# =============================================================================
# This script provides the main entry point for running the web application.
# 
# Usage:
#   python main.py              # Start web UI
#   python main.py --cli        # Run CLI batch test
#   python main.py --help       # Show help
# =============================================================================

import argparse
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import create_app, main as run_web_app


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO

    # Some libraries (or application modules imported earlier) may have
    # already configured the root logger. logging.basicConfig is a no-op
    # if handlers are already present. To reliably ensure a FileHandler is
    # attached, remove any existing root handlers and add our own handlers.
    root = logging.getLogger()

    # Remove and close existing handlers to avoid duplicate outputs
    if root.handlers:
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
                h.close()
            except Exception:
                # ignore handler cleanup errors
                pass

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    # File handler (explicit encoding for portability)
    fh = logging.FileHandler('log.log', encoding='utf-8')
    fh.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(sh)
    root.addHandler(fh)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt Injection Security Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Start web UI on port 7860
  python main.py --port 8080     # Start on custom port
  python main.py --share         # Create public Gradio link
        """
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for web UI (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio share link'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("Prompt Injection Security Research")
    
    app, custom_css = create_app()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=custom_css
    )


if __name__ == "__main__":
    main()
