#!/usr/bin/env python3
"""
Test script for the Coastline Angle Control widget

This script demonstrates the custom circular coastline angle selector.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt6.QtCore import Qt

from townsim.modules.ui.coastline_angle_control import CoastlineAngleControl


class TestWindow(QMainWindow):
    """Test window for the coastline angle control."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coastline Angle Control Test")
        self.setGeometry(100, 100, 400, 500)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Coastline Angle Control Test")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Click and drag around the circle to set the coastline angle.\n"
            "The red arrow shows the direction, and the shoreline (yellow line)\n" 
            "is perpendicular to the arrow direction."
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("margin: 10px;")
        layout.addWidget(instructions)
        
        # Coastline angle control
        self.angle_control = CoastlineAngleControl()
        self.angle_control.setAngle(41.7)  # Match the example image
        self.angle_control.angleChanged.connect(self.on_angle_changed)
        layout.addWidget(self.angle_control, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Angle display
        self.angle_label = QLabel(f"Current Angle: {self.angle_control.angle():.1f}°")
        self.angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angle_label.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(self.angle_label)
        
        # Test buttons
        button_layout = QVBoxLayout()
        
        test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for angle in test_angles:
            button = QPushButton(f"Set {angle}°")
            button.clicked.connect(lambda checked, a=angle: self.set_test_angle(a))
            button_layout.addWidget(button)
        
        layout.addLayout(button_layout)
        
        # Stretch to center everything
        layout.addStretch()
    
    def on_angle_changed(self, angle: float):
        """Handle angle changes from the control."""
        self.angle_label.setText(f"Current Angle: {angle:.1f}°")
        print(f"Coastline angle changed to: {angle:.1f}°")
    
    def set_test_angle(self, angle: float):
        """Set a test angle."""
        self.angle_control.setAngle(angle)


def main():
    """Run the test application."""
    app = QApplication(sys.argv)
    
    window = TestWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 