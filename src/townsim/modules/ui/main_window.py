"""
Main Window for TownSim Python

This module provides the main GUI window for the TownSim application.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QMenuBar, QStatusBar, QTabWidget, QSplitter,
    QFrame, QLabel, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon

from ..utils.logging_setup import get_logger


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing main window")
        
        self.setWindowTitle("TownSim Python - Town/City Generator & Simulator")
        self.setMinimumSize(1200, 800)
        
        # Initialize UI components
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        
        # Initialize simulation state
        self.current_simulation = None
        self.simulation_running = False
        
        self.logger.info("Main window initialized successfully")
    
    def _setup_ui(self) -> None:
        """Set up the main user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel (controls)
        self._setup_control_panel(main_splitter)
        
        # Right panel (map view)
        self._setup_map_view(main_splitter)
        
        # Set splitter proportions
        main_splitter.setSizes([300, 900])
    
    def _setup_control_panel(self, parent: QSplitter) -> None:
        """Set up the control panel on the left side."""
        control_panel = QFrame()
        control_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        control_panel.setMaximumWidth(350)
        
        layout = QVBoxLayout(control_panel)
        
        # Terrain Generation Group
        terrain_group = QGroupBox("Terrain Generation")
        terrain_layout = QVBoxLayout(terrain_group)
        
        # Generator type selection
        from PyQt6.QtWidgets import QComboBox
        self.generator_combo = QComboBox()
        self.generator_combo.addItem("Advanced Generator (Recommended)", "advanced")
        self.generator_combo.addItem("Basic Generator", "basic")
        terrain_layout.addWidget(QLabel("Generator Type:"))
        terrain_layout.addWidget(self.generator_combo)
        
        # Coastal type selection (for advanced generator)
        from ..terrain.advanced_terrain_generator import CoastalType
        self.coastal_combo = QComboBox()
        self.coastal_combo.addItem("Random", CoastalType.RANDOM)
        self.coastal_combo.addItem("Landlocked", CoastalType.LANDLOCKED)
        self.coastal_combo.addItem("Coastal", CoastalType.COASTAL)
        self.coastal_combo.addItem("Island", CoastalType.ISLAND)
        terrain_layout.addWidget(QLabel("Coastal Type:"))
        terrain_layout.addWidget(self.coastal_combo)
        
        # Coastline angle control (for coastal type)
        from .coastline_angle_control import CoastlineAngleControl
        self.coastline_angle_control = CoastlineAngleControl()
        self.coastline_angle_control.setAngle(45.0)  # Default angle
        self.coastline_angle_control.angleChanged.connect(self._on_coastline_angle_changed)
        self.coastal_combo.currentTextChanged.connect(self._on_coastal_type_changed)
        
        # Add the control with proper layout
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Coastline Angle:"))
        angle_layout.addStretch()
        terrain_layout.addLayout(angle_layout)
        terrain_layout.addWidget(self.coastline_angle_control)
        
        # Initially hide the angle control (only show for coastal type)
        self.coastline_angle_control.setVisible(False)
        
        # Terrain smoothness slider
        from PyQt6.QtWidgets import QSlider, QSpinBox
        self.smoothness_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothness_slider.setRange(0, 100)
        self.smoothness_slider.setValue(30)  # Default 0.3
        self.smoothness_label = QLabel("Terrain Smoothness: 0.3")
        self.smoothness_slider.valueChanged.connect(self._update_smoothness_label)
        terrain_layout.addWidget(self.smoothness_label)
        terrain_layout.addWidget(self.smoothness_slider)
        
        # Maximum lakes setting
        self.max_lakes_spinbox = QSpinBox()
        self.max_lakes_spinbox.setRange(0, 10)
        self.max_lakes_spinbox.setValue(2)  # Default 2
        terrain_layout.addWidget(QLabel("Maximum Lakes:"))
        terrain_layout.addWidget(self.max_lakes_spinbox)
        
        # Random seed setting
        from PyQt6.QtWidgets import QCheckBox, QLineEdit
        self.use_random_seed_checkbox = QCheckBox("Use Random Seed")
        self.use_random_seed_checkbox.setChecked(True)  # Default to random
        self.use_random_seed_checkbox.toggled.connect(self._toggle_seed_input)
        terrain_layout.addWidget(self.use_random_seed_checkbox)
        
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Enter seed (e.g. 42)")
        self.seed_input.setEnabled(False)  # Disabled when using random seed
        terrain_layout.addWidget(QLabel("Seed (for reproducible results):"))
        terrain_layout.addWidget(self.seed_input)
        
        # Generation buttons
        btn_generate_terrain = QPushButton("Generate Terrain")
        btn_generate_terrain.clicked.connect(self._generate_terrain)
        terrain_layout.addWidget(btn_generate_terrain)
        
        layout.addWidget(terrain_group)
        
        # Simulation Control Group
        sim_group = QGroupBox("Simulation Controls")
        sim_layout = QVBoxLayout(sim_group)
        
        # TODO: Add simulation controls
        sim_layout.addWidget(QLabel("Simulation controls will be added here"))
        
        layout.addWidget(sim_group)
        
        # Stretch to push everything to the top
        layout.addStretch()
        
        parent.addWidget(control_panel)
    
    def _setup_map_view(self, parent: QSplitter) -> None:
        """Set up the map view on the right side."""
        map_frame = QFrame()
        map_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout(map_frame)
        
        # TODO: Add actual map widget with zoom/pan capabilities
        placeholder_label = QLabel("Map view will be implemented here")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        
        layout.addWidget(placeholder_label)
        
        parent.addWidget(map_frame)
    
    def _setup_menu_bar(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Simulation", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_simulation)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Simulation", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_simulation)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Simulation", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_simulation)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self) -> None:
        """Set up the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def _update_smoothness_label(self, value: int) -> None:
        """Update the smoothness label when slider changes."""
        smoothness = value / 100.0
        self.smoothness_label.setText(f"Terrain Smoothness: {smoothness:.1f}")
    
    def _toggle_seed_input(self, use_random: bool) -> None:
        """Toggle the seed input field based on random seed checkbox."""
        self.seed_input.setEnabled(not use_random)
        if use_random:
            self.seed_input.clear()
    
    def _on_coastal_type_changed(self, text: str) -> None:
        """Handle coastal type selection changes."""
        # Show/hide the coastline angle control based on coastal type
        is_coastal = text == "Coastal"
        self.coastline_angle_control.setVisible(is_coastal)
        
        if is_coastal:
            self.logger.debug(f"Coastal type selected - showing angle control (current: {self.coastline_angle_control.angle():.1f}°)")
        else:
            self.logger.debug(f"Non-coastal type selected: {text} - hiding angle control")
    
    def _on_coastline_angle_changed(self, angle: float) -> None:
        """Handle coastline angle changes."""
        self.logger.debug(f"Coastline angle changed to: {angle:.1f}°")
    
    def _generate_terrain(self) -> None:
        """Generate new terrain."""
        self.logger.info("Generating terrain...")
        self.status_bar.showMessage("Generating terrain...")
        
        try:
            self.logger.debug("Starting terrain generation process...")
            
            # Get parameters from UI
            generator_type = self.generator_combo.currentData()
            coastal_type = self.coastal_combo.currentData()
            smoothness = self.smoothness_slider.value() / 100.0
            max_lakes = self.max_lakes_spinbox.value()
            
            self.logger.debug(f"Parameters: type={generator_type}, coastal={coastal_type}, smoothness={smoothness}, lakes={max_lakes}")
            
            # Determine random seed
            if self.use_random_seed_checkbox.isChecked():
                import random
                random_seed = random.randint(1, 999999)
                self.logger.info(f"Using random seed: {random_seed}")
            else:
                try:
                    random_seed = int(self.seed_input.text()) if self.seed_input.text() else 42
                except ValueError:
                    random_seed = 42
                    self.logger.warning("Invalid seed input, using default seed 42")
            
            if generator_type == "advanced":
                self.logger.debug("Importing advanced generator...")
                # Use advanced generator
                from ..terrain.advanced_terrain_generator import (
                    AdvancedTerrainGenerator, AdvancedTerrainParameters
                )
                
                self.logger.debug("Creating advanced parameters...")
                params = AdvancedTerrainParameters(
                    map_size=512,
                    random_seed=random_seed,
                    coastal_type=coastal_type,
                    terrain_smoothness=smoothness,
                    max_lakes=max_lakes,
                    noise_scale=5.0,
                    elevation_variance=0.7
                )
                
                # If coastal type is selected, use the angle from the control
                if coastal_type == coastal_type:
                    coastline_angle = self.coastline_angle_control.angle()
                    self.logger.debug(f"Using coastline angle: {coastline_angle:.1f}°")
                    # Store the angle for use in terrain generation
                    params.coastline_angle = coastline_angle
                
                self.logger.debug("Creating advanced generator instance...")
                generator = AdvancedTerrainGenerator()
                
                self.logger.debug("Calling generate_terrain...")
                terrain_map = generator.generate_terrain(params)
                
                self.logger.info(f"Advanced terrain generated successfully")
                seed_msg = f" (seed: {random_seed})" if self.use_random_seed_checkbox.isChecked() else ""
                self.status_bar.showMessage(f"Advanced terrain generated successfully!{seed_msg}")
                
            else:
                self.logger.debug("Using basic generator...")
                # Use basic generator
                from ..terrain.terrain_generator import TerrainGenerator
                from ..terrain.terrain_parameters import TerrainParameters
                
                params = TerrainParameters(
                    map_size=512,
                    random_seed=random_seed,
                    noise_scale=5.0,
                    elevation_variance=0.7
                )
                
                generator = TerrainGenerator()
                terrain_map = generator.generate_terrain(params)
                
                self.logger.info(f"Basic terrain generated successfully")
                seed_msg = f" (seed: {random_seed})" if self.use_random_seed_checkbox.isChecked() else ""
                self.status_bar.showMessage(f"Basic terrain generated successfully!{seed_msg}")
            
            self.logger.debug("Saving terrain preview...")
            # TODO: Display terrain in map view
            # For now, just save to file for inspection
            self._save_terrain_preview(terrain_map)
            self.logger.debug("Terrain generation process completed successfully")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Terrain generation failed: {e}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            self.status_bar.showMessage(f"Terrain generation failed: {e}")
            
            # Try to show a more user-friendly error message
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Terrain Generation Error",
                f"Failed to generate terrain:\n\n{str(e)}\n\nCheck the logs for more details."
            )
    
    def _save_terrain_preview(self, terrain_map) -> None:
        """Save terrain preview to file."""
        try:
            from PIL import Image
            import numpy as np
            from ..terrain.terrain_types import TerrainType
            
            # Create RGB image
            size = terrain_map.tiles.shape[0]
            img_array = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Color mapping for terrain types
            colors = {
                TerrainType.GRASS: (34, 139, 34),
                TerrainType.MUD: (139, 69, 19),
                TerrainType.SAND: (238, 203, 173),
                TerrainType.ROCK: (105, 105, 105),
                TerrainType.COAST: (255, 218, 185),
                TerrainType.SEA: (70, 130, 180),
                TerrainType.SNOW: (255, 250, 250)
            }
            
            # Apply colors
            for terrain_type, color in colors.items():
                mask = terrain_map.tiles == terrain_type
                img_array[mask] = color
            
            # Save image
            img = Image.fromarray(img_array)
            img.save("terrain_preview.png")
            
            self.logger.info("Terrain preview saved as terrain_preview.png")
            
        except Exception as e:
            self.logger.error(f"Failed to save terrain preview: {e}")
    
    def _new_simulation(self) -> None:
        """Start a new simulation."""
        self.logger.info("Starting new simulation")
        self.status_bar.showMessage("New simulation started")
    
    def _open_simulation(self) -> None:
        """Open an existing simulation."""
        self.logger.info("Opening simulation")
        self.status_bar.showMessage("Opening simulation - Coming soon!")
    
    def _save_simulation(self) -> None:
        """Save the current simulation."""
        self.logger.info("Saving simulation")
        self.status_bar.showMessage("Saving simulation - Coming soon!")
    
    def _zoom_in(self) -> None:
        """Zoom in on the map."""
        self.logger.debug("Zooming in")
        self.status_bar.showMessage("Zoom in - Coming soon!")
    
    def _zoom_out(self) -> None:
        """Zoom out on the map."""
        self.logger.debug("Zooming out")
        self.status_bar.showMessage("Zoom out - Coming soon!")
    
    def _show_about(self) -> None:
        """Show the about dialog."""
        from PyQt6.QtWidgets import QMessageBox
        
        QMessageBox.about(
            self,
            "About TownSim Python",
            "TownSim Python v0.1.0\n\n"
            "A town/city generator and growth simulator.\n\n"
            "Features:\n"
            "• Terrain generation with configurable parameters\n"
            "• Logical city layout generation\n"
            "• Time-based growth simulation\n"
            "• Event tracking and data export"
        )
    
    def load_simulation(self, filepath: str) -> None:
        """
        Load a simulation from file.
        
        Args:
            filepath: Path to the simulation file
        """
        self.logger.info(f"Loading simulation from: {filepath}")
        self.status_bar.showMessage(f"Loading simulation: {filepath}")
        
        # TODO: Implement simulation loading
        self.status_bar.showMessage("Simulation loading - Coming soon!") 