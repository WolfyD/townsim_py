"""
Coastline Angle Control Widget

A custom circular control for selecting coastline angles with visual representation
of water (blue), land (green), and shoreline (sandy yellow).
"""

import math
from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygon


class CoastlineAngleControl(QWidget):
    """Custom circular control for selecting coastline angles."""
    
    # Signal emitted when angle changes
    angleChanged = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Control properties
        self._angle = 45.0  # Current angle in degrees (0 = North, 90 = East)
        self._dragging = False
        self._radius = 80  # Control radius
        
        # Colors
        self._water_color = QColor(70, 130, 180)      # Steel blue
        self._land_color = QColor(34, 139, 34)        # Forest green  
        self._shore_color = QColor(238, 203, 173)     # Sandy yellow
        self._arrow_color = QColor(220, 20, 60)       # Crimson red
        self._text_color = QColor(0, 0, 0)            # Black
        self._marker_color = QColor(60, 60, 60)       # Dark gray
        
        # Setup widget
        self.setMinimumSize(200, 200)
        self.setMaximumSize(250, 250)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def angle(self) -> float:
        """Get the current angle in degrees."""
        return self._angle
    
    def setAngle(self, angle: float) -> None:
        """Set the angle in degrees (0-360)."""
        # Normalize angle to 0-360 range
        angle = angle % 360.0
        
        if abs(self._angle - angle) > 0.1:  # Avoid unnecessary updates
            self._angle = angle
            self.update()
            self.angleChanged.emit(self._angle)
    
    def paintEvent(self, event):
        """Paint the circular coastline control."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate center and radius
        rect = self.rect()
        center = QPoint(rect.width() // 2, rect.height() // 2)
        radius = min(rect.width(), rect.height()) // 2 - 20  # Leave margin
        self._radius = radius
        
        # Draw the main circle background (water)
        painter.setBrush(QBrush(self._water_color))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawEllipse(center.x() - radius, center.y() - radius, 
                          radius * 2, radius * 2)
        
        # Calculate angle for coastline
        # UI: 0°=North, 90°=East (clockwise from North)
        # Convert to standard math angle where 0°=East, 90°=North (counter-clockwise from East)
        math_angle = math.radians(90 - self._angle)
        
        # Create the land area (sector from coastline)
        self._draw_land_sector(painter, center, radius, math_angle)
        
        # Draw shoreline
        self._draw_shoreline(painter, center, radius, math_angle)
        
        # Draw degree markers
        self._draw_degree_markers(painter, center, radius)
        
        # Draw compass indicators
        self._draw_compass_indicators(painter, center, radius)
        
        # Draw direction arrow
        self._draw_direction_arrow(painter, center, radius, math_angle)
        
        # Draw center angle text
        self._draw_center_text(painter, center)
    
    def _draw_land_sector(self, painter: QPainter, center: QPoint, radius: int, math_angle: float):
        """Draw the land sector based on coastline angle."""
        painter.setBrush(QBrush(self._land_color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Simple approach: draw a sector in the direction the arrow points
        # The sector represents the land area
        
        # Convert UI angle back to get the land direction
        # UI angle 0° = North, 90° = East
        land_direction_deg = self._angle
        
        # Create a sector that spans ±90° around the land direction
        # This will be exactly half the circle
        start_angle_deg = land_direction_deg - 90
        end_angle_deg = land_direction_deg + 90
        
        # Draw the sector using Qt's drawPie method
        # Qt angles are in 16ths of a degree and start from 3 o'clock going clockwise
        # Convert our angles to Qt format
        qt_start_angle = int((90 - end_angle_deg) * 16)  # Convert to Qt's coordinate system
        qt_span_angle = int(180 * 16)  # 180 degrees span
        
        # Draw the pie slice
        pie_rect = QRect(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        painter.drawPie(pie_rect, qt_start_angle, qt_span_angle)
    
    def _draw_shoreline(self, painter: QPainter, center: QPoint, radius: int, math_angle: float):
        """Draw the shoreline between water and land."""
        painter.setPen(QPen(self._shore_color, 4))
        
        # The shoreline is perpendicular to the land direction (arrow)
        # If land points at angle θ, shoreline is perpendicular to it
        land_direction_deg = self._angle
        
        # Calculate perpendicular direction to the arrow
        # Arrow direction in drawing coordinates
        arrow_cos = math.sin(math.radians(land_direction_deg))  # X component
        arrow_sin = math.cos(math.radians(land_direction_deg))  # Y component
        
        # Perpendicular to arrow (90° rotation)
        perp_cos = -arrow_sin  # Perpendicular X
        perp_sin = arrow_cos   # Perpendicular Y
        
        # Calculate coastline endpoints
        x1 = center.x() + perp_cos * radius
        y1 = center.y() - perp_sin * radius  # Negative because Qt Y is flipped
        x2 = center.x() - perp_cos * radius
        y2 = center.y() + perp_sin * radius
        
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
    
    def _draw_degree_markers(self, painter: QPainter, center: QPoint, radius: int):
        """Draw degree markers every 5 degrees."""
        painter.setPen(QPen(self._marker_color, 1))
        
        for angle_deg in range(0, 360, 5):
            angle_rad = math.radians(angle_deg)
            
            # Determine marker length (longer for every 30 degrees)
            if angle_deg % 30 == 0:
                inner_radius = radius - 15
                line_width = 2
            else:
                inner_radius = radius - 8
                line_width = 1
            
            painter.setPen(QPen(self._marker_color, line_width))
            
            # Calculate marker position
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            x1 = center.x() + cos_a * inner_radius
            y1 = center.y() - sin_a * inner_radius
            x2 = center.x() + cos_a * radius
            y2 = center.y() - sin_a * radius
            
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
    
    def _draw_compass_indicators(self, painter: QPainter, center: QPoint, radius: int):
        """Draw compass indicators (N, E, S, W)."""
        painter.setPen(QPen(self._text_color, 2))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        # Compass directions: N=0°, E=90°, S=180°, W=270°
        compass_points = [
            (0, "N"),
            (90, "E"), 
            (180, "S"),
            (270, "W")
        ]
        
        for angle_deg, label in compass_points:
            angle_rad = math.radians(angle_deg)
            
            # Position text outside the circle
            text_radius = radius + 15
            x = center.x() + math.cos(angle_rad) * text_radius
            y = center.y() - math.sin(angle_rad) * text_radius
            
            # Center the text
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label)
            text_height = metrics.height()
            
            text_x = x - text_width // 2
            text_y = y + text_height // 4  # Adjust for baseline
            
            painter.drawText(int(text_x), int(text_y), label)
    
    def _draw_direction_arrow(self, painter: QPainter, center: QPoint, radius: int, math_angle: float):
        """Draw the direction arrow indicating land direction."""
        painter.setBrush(QBrush(self._arrow_color))
        painter.setPen(QPen(self._arrow_color, 2))
        
        # Arrow points in the direction of the land
        # Convert UI angle to drawing coordinates
        land_direction_deg = self._angle
        land_direction_rad = math.radians(land_direction_deg)
        
        arrow_length = radius + 10
        
        # Calculate arrow direction (0° = North/up, 90° = East/right)
        cos_a = math.sin(land_direction_rad)  # X component
        sin_a = math.cos(land_direction_rad)  # Y component (flipped for Qt coordinates)
        
        # Arrow tip
        tip_x = center.x() + cos_a * arrow_length
        tip_y = center.y() - sin_a * arrow_length  # Negative for Qt Y-up
        
        # Arrow base
        base_x = center.x() + cos_a * (radius - 5)
        base_y = center.y() - sin_a * (radius - 5)
        
        # Arrow sides (perpendicular to direction)
        perp_cos = -sin_a
        perp_sin = cos_a
        arrow_width = 6
        
        side1_x = base_x + perp_cos * arrow_width
        side1_y = base_y - perp_sin * arrow_width
        side2_x = base_x - perp_cos * arrow_width
        side2_y = base_y + perp_sin * arrow_width
        
        # Draw arrow
        arrow_points = [
            QPoint(int(tip_x), int(tip_y)),
            QPoint(int(side1_x), int(side1_y)),
            QPoint(int(side2_x), int(side2_y))
        ]
        
        arrow_polygon = QPolygon(arrow_points)
        painter.drawPolygon(arrow_polygon)
    
    def _draw_center_text(self, painter: QPainter, center: QPoint):
        """Draw the current angle value in the center."""
        painter.setPen(QPen(self._text_color))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        # Format angle text
        angle_text = f"{self._angle:.1f}°"
        
        # Center the text
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(angle_text)
        text_height = metrics.height()
        
        text_x = center.x() - text_width // 2
        text_y = center.y() + text_height // 4
        
        painter.drawText(text_x, text_y, angle_text)
    
    def mousePressEvent(self, event):
        """Handle mouse press for starting drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if click is within the control area
            center = QPoint(self.width() // 2, self.height() // 2)
            click_pos = event.position().toPoint()
            distance = math.sqrt((click_pos.x() - center.x())**2 + (click_pos.y() - center.y())**2)
            
            if distance <= self._radius + 20:  # Allow some margin
                self._dragging = True
                self._update_angle_from_mouse(event.position().toPoint())
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if self._dragging:
            self._update_angle_from_mouse(event.position().toPoint())
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to end drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
    
    def _update_angle_from_mouse(self, mouse_pos: QPoint):
        """Update angle based on mouse position."""
        center = QPoint(self.width() // 2, self.height() // 2)
        
        # Calculate angle from center to mouse position
        dx = mouse_pos.x() - center.x()
        dy = mouse_pos.y() - center.y()
        
        # Convert to angle (math coordinates: 0° = East, 90° = North)
        math_angle = math.atan2(-dy, dx)  # Negative dy because Qt Y is flipped
        
        # Convert to UI coordinates (0° = North, 90° = East)
        ui_angle = (90 - math.degrees(math_angle)) % 360
        
        self.setAngle(ui_angle) 