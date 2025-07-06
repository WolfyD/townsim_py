"""
Main Window for TownSim Python

This module provides the main GUI window for the TownSim application.
"""

import logging
from pathlib import Path
from typing import Optional
from ..terrain.advanced_terrain_generator import CoastalType

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QMenuBar, QStatusBar, QTabWidget, QSplitter,
    QFrame, QLabel, QPushButton, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon, QPixmap, QImage

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
        
        # Terrain type selection (flat/hilly/mountainous)
        self.terrain_type_combo = QComboBox()
        self.terrain_type_combo.addItem("Flat Plains", "flat")
        self.terrain_type_combo.addItem("Rolling Hills", "hilly")
        self.terrain_type_combo.addItem("Mountainous", "mountainous")
        self.terrain_type_combo.addItem("Mixed Terrain", "mixed")
        self.terrain_type_combo.setCurrentIndex(1)  # Default to rolling hills
        terrain_layout.addWidget(QLabel("Terrain Type:"))
        terrain_layout.addWidget(self.terrain_type_combo)
        
        # Terrain smoothness slider
        from PyQt6.QtWidgets import QSlider, QSpinBox
        self.smoothness_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothness_slider.setRange(0, 100)
        self.smoothness_slider.setValue(30)  # Default 0.3
        self.smoothness_label = QLabel("Terrain Smoothness: 0.3")
        self.smoothness_slider.valueChanged.connect(self._update_smoothness_label)
        terrain_layout.addWidget(self.smoothness_label)
        terrain_layout.addWidget(self.smoothness_slider)
        
        # Elevation variance slider (for fine-tuning)
        self.elevation_variance_slider = QSlider(Qt.Orientation.Horizontal)
        self.elevation_variance_slider.setRange(10, 100)
        self.elevation_variance_slider.setValue(50)  # Default 0.5
        self.elevation_variance_label = QLabel("Elevation Variance: 0.5")
        self.elevation_variance_slider.valueChanged.connect(self._update_elevation_variance_label)
        terrain_layout.addWidget(self.elevation_variance_label)
        terrain_layout.addWidget(self.elevation_variance_slider)
        
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
        
        # Map display widget
        from PyQt6.QtWidgets import QScrollArea
        
        # Create scroll area for map
        self.map_scroll_area = QScrollArea()
        self.map_scroll_area.setWidgetResizable(True)
        self.map_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create map display label
        self.map_display_label = QLabel()
        self.map_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_display_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.map_display_label.setText("Generate terrain to see map preview")
        self.map_display_label.setMinimumSize(400, 400)
        
        # Set up scroll area
        self.map_scroll_area.setWidget(self.map_display_label)
        layout.addWidget(self.map_scroll_area)
        
        # Add map controls
        controls_layout = QHBoxLayout()
        
        # Zoom controls
        from PyQt6.QtWidgets import QPushButton
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_fit_btn = QPushButton("Fit to Window")
        
        self.zoom_in_btn.clicked.connect(self._zoom_in_map)
        self.zoom_out_btn.clicked.connect(self._zoom_out_map)
        self.zoom_fit_btn.clicked.connect(self._zoom_fit_map)
        
        controls_layout.addWidget(self.zoom_in_btn)
        controls_layout.addWidget(self.zoom_out_btn)
        controls_layout.addWidget(self.zoom_fit_btn)
        controls_layout.addStretch()
        
        # Map info label
        self.map_info_label = QLabel("No terrain generated")
        controls_layout.addWidget(self.map_info_label)
        
        layout.addLayout(controls_layout)
        
        # Initialize zoom level
        self.current_zoom = 1.0
        self.original_pixmap = None
        
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
    
    def _update_elevation_variance_label(self, value: int) -> None:
        """Update the elevation variance label when slider changes."""
        variance = value / 100.0
        self.elevation_variance_label.setText(f"Elevation Variance: {variance:.1f}")
    
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
    
    def _configure_terrain_type_parameters(self, params, terrain_type: str) -> None:
        """Configure terrain parameters based on terrain type selection."""
        # Configure noise parameters based on terrain type
        if terrain_type == "flat":
            # Flat plains - minimal elevation variation
            params.noise_amplitudes = [0.3, 0.15, 0.05, 0.02]  # Reduced amplitudes
            params.elevation_exponent = 0.8  # Flatten valleys and peaks
            params.noise_scale = 3.0  # Larger features
            
        elif terrain_type == "hilly":
            # Rolling hills - moderate elevation variation
            params.noise_amplitudes = [1.0, 0.5, 0.25, 0.125]  # Standard amplitudes
            params.elevation_exponent = 1.2  # Moderate redistribution
            params.noise_scale = 5.0  # Medium features
            
        elif terrain_type == "mountainous":
            # Mountainous - high elevation variation
            params.noise_amplitudes = [1.5, 0.8, 0.4, 0.2]  # Increased amplitudes
            params.elevation_exponent = 1.8  # Sharp peaks and valleys
            params.noise_scale = 7.0  # Dramatic features
            
        elif terrain_type == "mixed":
            # Mixed terrain - varied elevation
            params.noise_amplitudes = [1.2, 0.6, 0.3, 0.15]  # Varied amplitudes
            params.elevation_exponent = 1.4  # Moderate-high redistribution
            params.noise_scale = 6.0  # Varied features
        
        self.logger.debug(f"Configured terrain type '{terrain_type}': amplitudes={params.noise_amplitudes}, exponent={params.elevation_exponent}")
    
    def _generate_terrain(self) -> None:
        """Generate new terrain."""
        self.logger.info("Generating terrain...")
        self.status_bar.showMessage("Generating terrain...")
        
        try:
            self.logger.debug("Starting terrain generation process...")
            
            # Get parameters from UI
            generator_type = self.generator_combo.currentData()
            coastal_type = self.coastal_combo.currentData()
            terrain_type = self.terrain_type_combo.currentData()
            smoothness = self.smoothness_slider.value() / 100.0
            elevation_variance = self.elevation_variance_slider.value() / 100.0
            max_lakes = self.max_lakes_spinbox.value()
            
            self.logger.debug(f"Parameters: type={generator_type}, coastal={coastal_type}, terrain={terrain_type}, smoothness={smoothness}, variance={elevation_variance}, lakes={max_lakes}")
            
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
                    elevation_variance=elevation_variance
                )
                
                # Set terrain type-specific parameters
                params.terrain_type = terrain_type
                self._configure_terrain_type_parameters(params, terrain_type)
                
                # If coastal type is selected, use the angle from the control
                if coastal_type == CoastalType.COASTAL:
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
            
            self.logger.debug("Displaying terrain preview...")
            # Display terrain in map view
            self._display_terrain_preview(terrain_map)
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
    
    def _display_terrain_preview(self, terrain_map) -> None:
        """Display terrain preview in the UI and optionally save to file."""
        try:
            from PIL import Image
            import numpy as np
            from ..terrain.terrain_types import TerrainType
            
            # Create RGB image
            size = terrain_map.tiles.shape[0]
            img_array = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Enhanced color mapping for better visual distinction
            colors = {
                TerrainType.GRASS: (34, 139, 34),       # Forest green
                TerrainType.MUD: (139, 69, 19),         # Saddle brown
                TerrainType.SAND: (238, 203, 173),      # Peach puff
                TerrainType.ROCK: (105, 105, 105),      # Dim gray
                TerrainType.COAST: (255, 218, 185),     # Peach
                TerrainType.SEA: (70, 130, 180),        # Steel blue
                TerrainType.SNOW: (255, 250, 250)       # Snow white
            }
            
            # Apply colors
            for terrain_type, color in colors.items():
                mask = terrain_map.tiles == terrain_type
                img_array[mask] = color
            
            # Convert to QImage and then QPixmap for display
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create and store the pixmap
            self.original_pixmap = QPixmap.fromImage(q_image)
            
            # Reset zoom and display
            self.current_zoom = 1.0
            self._update_map_display()
            
            # Update map info
            total_tiles = size * size
            terrain_counts = {}
            for terrain_type in TerrainType:
                count = np.sum(terrain_map.tiles == terrain_type)
                if count > 0:
                    percentage = (count / total_tiles) * 100
                    terrain_counts[terrain_type.name] = percentage
            
            # Create info text
            info_parts = [f"{size}x{size}"]
            if 'SEA' in terrain_counts:
                info_parts.append(f"Water: {terrain_counts['SEA']:.1f}%")
            if 'GRASS' in terrain_counts:
                info_parts.append(f"Farmland: {terrain_counts['GRASS']:.1f}%")
            if 'ROCK' in terrain_counts:
                info_parts.append(f"Rocky: {terrain_counts['ROCK']:.1f}%")
            
            self.map_info_label.setText(" | ".join(info_parts))
            
            # Optional: Save to file for debugging
            # img = Image.fromarray(img_array)
            # img.save("terrain_preview.png")
            
            self.logger.info(f"Terrain preview displayed: {size}x{size} with {len(terrain_counts)} terrain types")
            
        except Exception as e:
            self.logger.error(f"Failed to display terrain preview: {e}")
            # Show error in UI
            self.map_display_label.setText(f"Error displaying terrain:\n{str(e)}")
            self.map_info_label.setText("Display error")
    
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
    
    def _zoom_in_map(self) -> None:
        """Zoom in on the terrain map."""
        if self.original_pixmap is not None:
            self.current_zoom = min(self.current_zoom * 1.25, 5.0)  # Max 5x zoom
            self._update_map_display()
            self.logger.debug(f"Zoomed in to {self.current_zoom:.2f}x")
    
    def _zoom_out_map(self) -> None:
        """Zoom out on the terrain map."""
        if self.original_pixmap is not None:
            self.current_zoom = max(self.current_zoom / 1.25, 0.1)  # Min 0.1x zoom
            self._update_map_display()
            self.logger.debug(f"Zoomed out to {self.current_zoom:.2f}x")
    
    def _zoom_fit_map(self) -> None:
        """Fit the terrain map to the window."""
        if self.original_pixmap is not None:
            # Calculate zoom to fit the scroll area
            scroll_size = self.map_scroll_area.size()
            pixmap_size = self.original_pixmap.size()
            
            zoom_x = scroll_size.width() / pixmap_size.width()
            zoom_y = scroll_size.height() / pixmap_size.height()
            self.current_zoom = min(zoom_x, zoom_y) * 0.9  # 90% to leave some margin
            
            self._update_map_display()
            self.logger.debug(f"Fit to window: {self.current_zoom:.2f}x")
    
    def _update_map_display(self) -> None:
        """Update the map display with current zoom level."""
        if self.original_pixmap is not None:
            # Scale the pixmap
            scaled_pixmap = self.original_pixmap.scaled(
                self.original_pixmap.size() * self.current_zoom,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Update the display
            self.map_display_label.setPixmap(scaled_pixmap)
            self.map_display_label.resize(scaled_pixmap.size())
    
    def _zoom_in(self) -> None:
        """Legacy zoom in method - redirects to map zoom."""
        self._zoom_in_map()
    
    def _zoom_out(self) -> None:
        """Legacy zoom out method - redirects to map zoom."""
        self._zoom_out_map()
    
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