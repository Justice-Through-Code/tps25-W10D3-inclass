import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import threading
import queue
import joblib
import os

class WeatherAlertSystem:
    """Manages weather alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'temperature_high': 90,
            'temperature_low': 32,
            'wind_speed': 25,
            'precipitation': 0.5
        }
        self.alert_callbacks = []
        
    def check_conditions(self, current_weather, forecast):
        """Check for alert conditions"""
        new_alerts = []
        
        # Temperature alerts
        if current_weather.temperature > self.thresholds['temperature_high']:
            new_alerts.append(WeatherAlert(
                type='HIGH_TEMP',
                severity='WARNING',
                message=f"High temperature alert: {current_weather.temperature:.0f}°F",
                timestamp=datetime.now()
            ))
            
        # Check forecast for upcoming conditions
        for f in forecast[:12]:  # Next 12 hours
            if f.temperature < self.thresholds['temperature_low']:
                new_alerts.append(WeatherAlert(
                    type='FREEZE',
                    severity='WARNING',
                    message=f"Freeze warning: {f.temperature:.0f}°F expected at {f.timestamp.strftime('%I %p')}",
                    timestamp=datetime.now()
                ))
                break
                
        # Notify callbacks
        for alert in new_alerts:
            self.trigger_alert(alert)
            
        return new_alerts
    
    def trigger_alert(self, alert):
        """Trigger alert callbacks"""
        self.alerts.append(alert)
        for callback in self.alert_callbacks:
            callback(alert)
    
    def register_callback(self, callback):
        """Register alert callback"""
        self.alert_callbacks.append(callback)

@dataclass
class WeatherAlert:
    type: str
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    timestamp: datetime

class AlertWidget(ttk.Frame):
    """Widget for displaying weather alerts"""
    
    def __init__(self, parent, alert_system):
        super().__init__(parent)
        self.alert_system = alert_system
        self.alert_system.register_callback(self.on_alert)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create alert UI"""
        # Alert container
        self.alert_frame = ttk.Frame(self)
        self.alert_frame.pack(fill='both', expand=True)
        
        # No alerts message
        self.no_alerts_label = ttk.Label(self.alert_frame, 
                                       text="No active weather alerts",
                                       font=('Arial', 12))
        self.no_alerts_label.pack(pady=20)
        
        self.alert_widgets = []
        
    def on_alert(self, alert):
        """Handle new alert"""
        # Remove no alerts message
        self.no_alerts_label.pack_forget()
        
        # Create alert widget
        alert_widget = self.create_alert_widget(alert)
        alert_widget.pack(fill='x', padx=10, pady=5)
        self.alert_widgets.append(alert_widget)
        
        # Limit displayed alerts
        if len(self.alert_widgets) > 5:
            oldest = self.alert_widgets.pop(0)
            oldest.destroy()
            
    def create_alert_widget(self, alert):
        """Create individual alert display"""
        frame = ttk.Frame(self.alert_frame, relief='solid', borderwidth=1)
        
        # Severity indicator
        severity_colors = {
            'INFO': '#2196F3',
            'WARNING': '#FFC107',
            'CRITICAL': '#F44336'
        }
        
        indicator = tk.Label(frame, text="⚠", font=('Arial', 16),
                           fg=severity_colors.get(alert.severity, '#000'))
        indicator.pack(side='left', padx=10)
        
        # Alert text
        text_frame = ttk.Frame(frame)
        text_frame.pack(side='left', fill='both', expand=True)
        
        ttk.Label(text_frame, text=alert.message,
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(text_frame, text=alert.timestamp.strftime('%I:%M %p'),
                 font=('Arial', 8)).pack(anchor='w')
        
        # Dismiss button
        ttk.Button(frame, text="✕", width=3,
                  command=lambda: self.dismiss_alert(frame)).pack(side='right', padx=5)
        
        return frame
    
    def dismiss_alert(self, alert_widget):
        """Dismiss an alert"""
        alert_widget.destroy()
        self.alert_widgets = [w for w in self.alert_widgets if w.winfo_exists()]
        
        if not self.alert_widgets:
            self.no_alerts_label.pack(pady=20)
# Location-Based Features:
# python
class LocationManager:
    """Manages location services"""
    
    def __init__(self):
        self.current_location = None
        self.saved_locations = []
        self.load_saved_locations()
        
    def get_current_location(self):
        """Get current location (simplified)"""
        # In real app, would use geolocation API
        return Location(
            name="Edison, NJ",
            latitude=40.5187,
            longitude=-74.4121,
            timezone="America/New_York"
        )
    
    def add_location(self, location):
        """Add a saved location"""
        self.saved_locations.append(location)
        self.save_locations()
        
    def save_locations(self):
        """Save locations to file"""
        data = [loc.__dict__ for loc in self.saved_locations]
        with open('locations.json', 'w') as f:
            json.dump(data, f)
            
    def load_saved_locations(self):
        """Load saved locations"""
        try:
            with open('locations.json', 'r') as f:
                data = json.load(f)
                self.saved_locations = [Location(**d) for d in data]
        except FileNotFoundError:
            self.saved_locations = []

@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str

class LocationSelector(ttk.Frame):
    """Widget for selecting locations"""
    
    def __init__(self, parent, location_manager, callback):
        super().__init__(parent)
        self.location_manager = location_manager
        self.callback = callback
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create location selector UI"""
        # Current location
        current_frame = ttk.LabelFrame(self, text="Current Location", padding=10)
        current_frame.pack(fill='x', padx=10, pady=5)
        
        current = self.location_manager.get_current_location()
        ttk.Label(current_frame, text=current.name,
                 font=('Arial', 12, 'bold')).pack()
        
        ttk.Button(current_frame, text="Use Current Location",
                  command=lambda: self.select_location(current)).pack(pady=5)
        
        # Saved locations
        saved_frame = ttk.LabelFrame(self, text="Saved Locations", padding=10)
        saved_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Location list
        self.location_listbox = tk.Listbox(saved_frame, height=5)
        self.location_listbox.pack(fill='both', expand=True)
        
        for loc in self.location_manager.saved_locations:
            self.location_listbox.insert(tk.END, loc.name)
            
        # Select button
        ttk.Button(saved_frame, text="Select",
                  command=self.on_select_saved).pack(pady=5)
        
        # Add location
        add_frame = ttk.Frame(self)
        add_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(add_frame, text="Add New Location",
                  command=self.add_location_dialog).pack()
    
    def select_location(self, location):
        """Select a location"""
        self.callback(location)
        
    def on_select_saved(self):
        """Handle saved location selection"""
        selection = self.location_listbox.curselection()
        if selection:
            index = selection[0]
            location = self.location_manager.saved_locations[index]
            self.select_location(location)
            
    def add_location_dialog(self):
        """Show add location dialog"""
        dialog = tk.Toplevel(self)
        dialog.title("Add Location")
        dialog.geometry("300x200")
        
        # Location name
        ttk.Label(dialog, text="Location Name:").grid(row=0, column=0, pady=10)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).grid(row=0, column=1)
        
        # Coordinates
        ttk.Label(dialog, text="Latitude:").grid(row=1, column=0, pady=5)
        lat_var = tk.DoubleVar()
        ttk.Entry(dialog, textvariable=lat_var).grid(row=1, column=1)
        
        ttk.Label(dialog, text="Longitude:").grid(row=2, column=0, pady=5)
        lon_var = tk.DoubleVar()
        ttk.Entry(dialog, textvariable=lon_var).grid(row=2, column=1)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        def save_location():
            location = Location(
                name=name_var.get(),
                latitude=lat_var.get(),
                longitude=lon_var.get(),
                timezone="UTC"  # Simplified
            )
            self.location_manager.add_location(location)
            self.location_listbox.insert(tk.END, location.name)
            dialog.destroy()
            
        ttk.Button(button_frame, text="Save", command=save_location).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='left')
# Export and Sharing Features:
# python
class ExportManager:
    """Handles data export and sharing"""
    
    def __init__(self, app):
        self.app = app
        
    def export_forecast_image(self, filename=None):
        """Export forecast as image"""
        if not filename:
            filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
        # Create figure with forecast
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Temperature forecast
        historical = self.app.data_manager.get_historical_data(24)
        forecasts = self.app.prediction_engine.predict_ensemble(None, 48)
        
        # Plot historical
        ax1.plot(historical['timestamp'], historical['temperature'], 
                'b-', linewidth=2, label='Historical')
        
        # Plot forecast
        forecast_times = [f.timestamp for f in forecasts]
        forecast_temps = [f.temperature for f in forecasts]
        ax1.plot(forecast_times, forecast_temps, 
                'r--', linewidth=2, label='Forecast')
        
        # Confidence bands
        upper = [f.temperature_high for f in forecasts]
        lower = [f.temperature_low for f in forecasts]
        ax1.fill_between(forecast_times, lower, upper, 
                        alpha=0.2, color='red')
        
        ax1.set_title('Temperature Forecast', fontsize=16)
        ax1.set_ylabel('Temperature (°F)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precipitation forecast
        precip = [f.precipitation_chance * 100 for f in forecasts]
        ax2.bar(range(len(precip)), precip, color='blue', alpha=0.6)
        ax2.set_title('Precipitation Probability', fontsize=16)
        ax2.set_ylabel('Chance (%)')
        ax2.set_xlabel('Hours Ahead')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def export_data_csv(self, filename=None):
        """Export forecast data as CSV"""
        if not filename:
            filename = f"forecast_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        # Get forecast data
        forecasts = self.app.prediction_engine.predict_ensemble(None, 48)
        
        # Create DataFrame
        data = []
        for f in forecasts:
            data.append({
                'timestamp': f.timestamp,
                'temperature': f.temperature,
                'temperature_low': f.temperature_low,
                'temperature_high': f.temperature_high,
                'humidity': f.humidity,
                'precipitation_chance': f.precipitation_chance,
                'conditions': f.conditions,
                'confidence': f.confidence
            })
            
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        return filename
    
    def generate_report(self):
        """Generate weather report"""
        report = []
        report.append("WEATHER FORECAST REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current conditions
        current = self.app.data_manager.current_data
        if current:
            report.append("CURRENT CONDITIONS")
            report.append("-" * 20)
            report.append(f"Temperature: {current.temperature:.1f}°F")
            report.append(f"Conditions: {current.conditions}")
            report.append(f"Humidity: {current.humidity:.0f}%")
            report.append(f"Wind: {current.wind_speed:.0f} mph")
            report.append("")
        
        # Forecast summary
        forecasts = self.app.prediction_engine.predict_ensemble(None, 48)
        
        report.append("24-HOUR FORECAST")
        report.append("-" * 20)
        
        # Find highs and lows
        temps_24h = [f.temperature for f in forecasts[:24]]
        high_temp = max(temps_24h)
        low_temp = min(temps_24h)
        
        report.append(f"High: {high_temp:.0f}°F")
        report.append(f"Low: {low_temp:.0f}°F")
        
        # Precipitation chance
        max_precip = max(f.precipitation_chance for f in forecasts[:24])
        if max_precip > 0.1:
            report.append(f"Precipitation: {max_precip:.0%} chance")
            
        report.append("")
        
        # Hourly breakdown
        report.append("HOURLY BREAKDOWN")
        report.append("-" * 20)
        
        for i, f in enumerate(forecasts[:12]):  # Next 12 hours
            report.append(f"{f.timestamp.strftime('%I %p'):6} - "
                         f"{f.temperature:5.1f}°F - "
                         f"{f.conditions:15} - "
                         f"Rain: {f.precipitation_chance:4.0%}")
        
        return "\n".join(report)

class ExportDialog:
    """Export dialog for various formats"""
    
    def __init__(self, parent, export_manager):
        self.export_manager = export_manager
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Export Forecast")
        self.dialog.geometry("400x300")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create export UI"""
        # Format selection
        ttk.Label(self.dialog, text="Select Export Format:",
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Export options
        options = [
            ("Forecast Image (PNG)", self.export_image),
            ("Data Spreadsheet (CSV)", self.export_csv),
            ("Weather Report (TXT)", self.export_report),
            ("Complete Package (ZIP)", self.export_all)
        ]
        
        for text, command in options:
            frame = ttk.Frame(self.dialog)
            frame.pack(fill='x', padx=20, pady=5)
            
            ttk.Label(frame, text=text, width=25).pack(side='left')
            ttk.Button(frame, text="Export", command=command).pack(side='right')
        
        # Close button
        ttk.Button(self.dialog, text="Close",
                  command=self.dialog.destroy).pack(pady=20)
    
    def export_image(self):
        """Export as image"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            self.export_manager.export_forecast_image(filename)
            messagebox.showinfo("Success", f"Forecast saved to {filename}")
    
    def export_csv(self):
        """Export as CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            self.export_manager.export_data_csv(filename)
            messagebox.showinfo("Success", f"Data saved to {filename}")
    
    def export_report(self):
        """Export as text report"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            report = self.export_manager.generate_report()
            with open(filename, 'w') as f:
                f.write(report)
            messagebox.showinfo("Success", f"Report saved to {filename}")
    
    def export_all(self):
        """Export complete package"""
        directory = filedialog.askdirectory()
        
        if directory:
            import zipfile
            import os
            
            # Create temporary files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_filename = os.path.join(directory, f"weather_forecast_{timestamp}.zip")
            
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                # Export each format
                image_file = self.export_manager.export_forecast_image()
                zipf.write(image_file, "forecast.png")
                os.remove(image_file)
                
                csv_file = self.export_manager.export_data_csv()
                zipf.write(csv_file, "forecast_data.csv")
# Continue
# Edit
# python
                os.remove(csv_file)
               
               # Write report
            report = self.export_manager.generate_report()
            report_file = "forecast_report.txt"
            with open(report_file, 'w') as f:
                   f.write(report)
            zipf.write(report_file, report_file)
            os.remove(report_file)
           
        messagebox.showinfo("Success", f"Complete package saved to {zip_filename}")
