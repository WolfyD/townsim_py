#!/usr/bin/env python3
"""
TownSim Python - Main Entry Point

This is the main entry point for the TownSim Python application.
It handles command-line arguments and launches the appropriate interface.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from townsim.modules.utils.logging_setup import setup_logging
from townsim.modules.ui.main_window import MainWindow


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TownSim Python - Town/City Generator and Growth Simulator"
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line interface mode (GUI is default)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--load",
        type=str,
        help="Load a saved simulation file on startup"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main application entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else getattr(logging, args.log_level)
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting TownSim Python - CLI Mode: {args.cli}")
    
    try:
        if args.cli:
            # TODO: Implement CLI interface
            from townsim.modules.ui.cli_interface import run_cli
            return run_cli(args)
        else:
            # Launch GUI interface
            from PyQt6.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            app.setApplicationName("TownSim Python")
            app.setApplicationVersion("0.1.0")
            
            window = MainWindow()
            
            # Load file if specified
            if args.load:
                window.load_simulation(args.load)
            
            window.show()
            return app.exec()
            
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 