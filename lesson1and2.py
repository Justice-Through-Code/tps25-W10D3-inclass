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

@dataclass
# Defines a structured record for current weather conditions (temp, humidity, etc.).
class WeatherData:
    """Data class for weather information"""
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: int
    conditions: str
    
@dataclass
# Defines a record for predicted future weather, including confidence and temperature range.
class Forecast:
    """Data class for weather forecast"""
    timestamp: datetime
    temperature: float
    temperature_low: float
    temperature_high: float
    humidity: float
    precipitation_chance: float
    conditions: str
    confidence: float

class WeatherDataManager:
    
    """Manages weather data collection and storage"""
    
    def __init__(self):
        self.current_data = None
        self.historical_data = pd.DataFrame()
        self.forecasts = []
        self.data_queue = queue.Queue()
        
    def fetch_current_weather(self) -> Optional[WeatherData]:
        """Fetch current weather data"""
        # Simulate API call using np.random to generate weather data
        try:
            current = WeatherData(
                timestamp=datetime.now(),
                temperature=70 + np.random.normal(0, 5),
                humidity=60 + np.random.normal(0, 10),
                pressure=29.92 + np.random.normal(0, 0.5),
                wind_speed=10 + np.random.normal(0, 3),
                wind_direction=int(np.random.uniform(0, 360)),
                conditions=np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Partly Cloudy'])
            )
            self.current_data = current
            return current
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return None
    
    def get_historical_data(self, hours: int = 24) -> pd.DataFrame:
        """Get historical weather data"""
        # Generate or load historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Simulate historical data for plotting trends over the past 24 hours
        timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
        data = {
            'timestamp': timestamps,
            'temperature': 70 + 10 * np.sin(np.arange(len(timestamps)) * np.pi / 12) + 
                          np.random.normal(0, 2, len(timestamps)),
            'humidity': 60 + np.random.normal(0, 10, len(timestamps)),
            'pressure': 29.92 + np.random.normal(0, 0.5, len(timestamps))
        }
        
        self.historical_data = pd.DataFrame(data)
        return self.historical_data

class PredictionEngine:
    """Handles all prediction operations"""
    # Handles forcasting using ensemble modeling
    
    def __init__(self, models: Dict):
        self.models = models
        self.ensemble_weights = {'linear': 0.3, 'rf': 0.5, 'arima': 0.2}
        
    def predict_ensemble(self, features: pd.DataFrame, horizon: int) -> List[Forecast]:
        """Make ensemble predictions"""
        predictions = []
        # Generates fake predictions using multiple “models” (linear, rf, arima) 
        # and combines them with weighted averaging. Includes simulated confidence based on variance.
        for i in range(horizon):
            # Get predictions from each model
            model_predictions = {}
            
            for name, model in self.models.items():
                if name in self.ensemble_weights:
                    # Make prediction (simplified)
                    pred = 70 + 10 * np.sin((i + 12) * np.pi / 12) + np.random.normal(0, 2)
                    model_predictions[name] = pred
            
            # Weighted average
            ensemble_pred = sum(
                self.ensemble_weights[name] * pred 
                for name, pred in model_predictions.items()
            )
            
            # Calculate confidence (simplified)
            variance = np.var(list(model_predictions.values()))
            confidence = max(0.5, 1 - variance / 10)
            
            # Create forecast
            forecast = Forecast(
                timestamp=datetime.now() + timedelta(hours=i),
                temperature=ensemble_pred,
                temperature_low=ensemble_pred - 5,
                temperature_high=ensemble_pred + 5,
                humidity=60 + np.random.normal(0, 10),
                precipitation_chance=np.random.uniform(0, 0.3),
                conditions='Partly Cloudy',
                confidence=confidence
            )
            
            predictions.append(forecast)
        
        return predictions

class WeatherPredictorApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Predictor Pro")
        self.root.geometry("1200x800")
        
        # Apply modern styling
        self.setup_styles()
        
        # Initialize components
        self.data_manager = WeatherDataManager()
        self.prediction_engine = PredictionEngine({})  # Load models here
        self.location_manager = LocationManager()
        self.export_manager = ExportManager(self)
        
        # Setup UI
        self.setup_ui()
        
        # Start data updates
        self.update_current_weather()

        self.alert_system = WeatherAlertSystem()  # Create the alert system
        self.alert_widget = AlertWidget(self.root, self.alert_system)  # Create the UI widget
        self.alert_widget.grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        
    def setup_styles(self):
        """Configure application styling"""
        style = ttk.Style()
        
        # Configure colors
        self.colors = {
            'bg': '#f0f0f0',
            'primary': '#2196F3',
            'secondary': '#FFC107',
            'success': '#4CAF50',
            'danger': '#F44336',
            'text': '#212121',
            'text_light': '#757575'
        }
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 12))
        
    def setup_ui(self):
        """Create the main user interface"""
        # Create main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=1)
        
        # Create sections
        self.create_header(main_container)
        self.create_current_weather_panel(main_container)
        self.create_forecast_panel(main_container)
        self.create_charts_panel(main_container)
        self.create_control_panel(main_container)
        

        #  Add alerts widget here
        # self.alert_system = WeatherAlertSystem()
        # self.alert_widget = AlertWidget(main_container, self.alert_system)
        # self.alert_widget.grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        
    def create_header(self, parent):
        """Create application header"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="Weather Predictor Pro", 
                               style='Title.TLabel')
        title_label.pack(side='left')
        
        # Last update time
        self.update_label = ttk.Label(header_frame, text="Last updated: Never", 
                                     style='Info.TLabel')
        self.update_label.pack(side='right')
        
    def create_current_weather_panel(self, parent):
        """Create current weather display"""
        current_frame = ttk.LabelFrame(parent, text="Current Conditions", padding="20")
        current_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Temperature display
        self.temp_label = ttk.Label(current_frame, text="--°F", 
                                   font=('Arial', 48, 'bold'))
        self.temp_label.grid(row=0, column=0, columnspan=2)
        
        # Conditions
        self.conditions_label = ttk.Label(current_frame, text="--", 
                                         font=('Arial', 18))
        self.conditions_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Details grid
        details = [
            ("Humidity:", "humidity_label"),
            ("Pressure:", "pressure_label"),
            ("Wind:", "wind_label"),
            ("Feels Like:", "feels_like_label")
        ]
        
        for i, (label_text, attr_name) in enumerate(details):
            ttk.Label(current_frame, text=label_text).grid(row=i+2, column=0, 
                                                          sticky='e', padx=(0, 10))
            label = ttk.Label(current_frame, text="--")
            label.grid(row=i+2, column=1, sticky='w')
            setattr(self, attr_name, label)

    def open_export_dialog(self):
        ExportDialog(self.root, self.export_manager)
    
    def create_forecast_panel(self, parent):
        """Create forecast display"""
        forecast_frame = ttk.LabelFrame(parent, text="Forecast", padding="10")
        forecast_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Forecast type selector
        control_frame = ttk.Frame(forecast_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(control_frame, text="Forecast Type:").pack(side='left', padx=(0, 10))
        
        self.forecast_type = tk.StringVar(value="hourly")
        forecast_types = [("Hourly", "hourly"), ("Daily", "daily"), ("Weekly", "weekly")]
        
        for text, value in forecast_types:
            ttk.Radiobutton(control_frame, text=text, variable=self.forecast_type,
                          value=value, command=self.update_forecast_display).pack(side='left', padx=5)
        
        # Forecast display area
        self.forecast_canvas = tk.Canvas(forecast_frame, height=300, bg='white')
        self.forecast_canvas.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(forecast_frame, orient='horizontal', 
                                command=self.forecast_canvas.xview)
        scrollbar.pack(fill='x')
        self.forecast_canvas.configure(xscrollcommand=scrollbar.set)
        
    def on_location_selected(self, location):
        print(f"Location selected: {location.name}")

    def create_charts_panel(self, parent):
        """Create charts section"""
        charts_frame = ttk.LabelFrame(parent, text="Trends", padding="10")
        charts_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                         pady=(10, 0))
        
        # Chart notebook
        self.chart_notebook = ttk.Notebook(charts_frame)
        self.chart_notebook.pack(fill='both', expand=True)

        self.location_selector_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.location_selector_frame, text="Locations")

        # Create the LocationSelector inside this tab
        self.location_selector = LocationSelector(
            self.location_selector_frame,
            self.location_manager,
            self.on_location_selected
        )
        self.location_selector.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Temperature trend
        self.temp_chart_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.temp_chart_frame, text="Temperature Trend")
        
        # Precipitation probability
        self.precip_chart_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.precip_chart_frame, text="Precipitation")
        
        # Initialize matplotlib figures
        self.setup_charts()
        
    def create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Update controls
        ttk.Button(control_frame, text="Update Now", 
                  command=self.manual_update).pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="Auto-update:").pack(side='left', padx=(20, 5))
        
        self.auto_update_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, variable=self.auto_update_var,
                       command=self.toggle_auto_update).pack(side='left')
        
        # Settings button
        ttk.Button(control_frame, text="Settings", 
                  command=self.show_settings).pack(side='right', padx=5)
        
        ttk.Button(control_frame, text="Export", command=self.open_export_dialog).pack(side='right', padx=5)

        # Status
        self.status_label = ttk.Label(control_frame, text="Ready", 
                                     foreground=self.colors['success'])
        self.status_label.pack(side='right', padx=20)
    
    def setup_charts(self):
        """Initialize matplotlib charts"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        # Temperature trend chart
        self.temp_figure = Figure(figsize=(10, 3))
        self.temp_ax = self.temp_figure.add_subplot(111)
        self.temp_canvas = FigureCanvasTkAgg(self.temp_figure, self.temp_chart_frame)
        self.temp_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Precipitation chart
        self.precip_figure = Figure(figsize=(10, 3))
        self.precip_ax = self.precip_figure.add_subplot(111)
        self.precip_canvas = FigureCanvasTkAgg(self.precip_figure, self.precip_chart_frame)
        self.precip_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def update_current_weather(self):
        """Update current weather display"""
        # Fetch current data
        current = self.data_manager.fetch_current_weather()
        
        if current:
            # Update display
            self.temp_label.config(text=f"{current.temperature:.0f}°F")
            self.conditions_label.config(text=current.conditions)
            self.humidity_label.config(text=f"{current.humidity:.0f}%")
            self.pressure_label.config(text=f"{current.pressure:.2f} in")
            self.wind_label.config(text=f"{current.wind_speed:.0f} mph")
            
            # Calculate feels like
            feels_like = current.temperature  # Simplified
            self.feels_like_label.config(text=f"{feels_like:.0f}°F")
            
            # Update timestamp
            self.update_label.config(text=f"Last updated: {datetime.now().strftime('%I:%M %p')}")
            
            # Update charts
            self.update_charts()
            
        # Schedule next update if auto-update is on
        if self.auto_update_var.get():
            self.root.after(60000, self.update_current_weather)  # Update every minute
    
    def update_forecast_display(self):
        """Update forecast based on selected type"""
        forecast_type = self.forecast_type.get()
        forecasts = self.prediction_engine.predict_ensemble(None, 24)

        # 🔔 Check for alerts
        if self.data_manager.current_data:
            self.alert_system.check_conditions(self.data_manager.current_data, forecasts)

        if forecast_type == "hourly":
            self.show_hourly_forecast()
        elif forecast_type == "daily":
            self.show_daily_forecast()
        else:
            self.show_weekly_forecast()
    
    def show_hourly_forecast(self):
        """Display hourly forecast"""
        # Clear canvas
        self.forecast_canvas.delete('all')
        
        # Get predictions
        forecasts = self.prediction_engine.predict_ensemble(None, 24)
        
        # Display parameters
        card_width = 80
        card_height = 250
        card_spacing = 10
        y_offset = 10
        
        # Create forecast cards
        for i, forecast in enumerate(forecasts[:24]):  # 24 hours
            x = i * (card_width + card_spacing) + card_spacing
            
            # Card background
            self.forecast_canvas.create_rectangle(
                x, y_offset, x + card_width, y_offset + card_height,
                fill='white', outline='#ddd'
            )
            
            # Time
            time_text = forecast.timestamp.strftime('%I %p')
            self.forecast_canvas.create_text(
                x + card_width/2, y_offset + 20,
                text=time_text, font=('Arial', 10, 'bold')
            )
            
            # Temperature
            self.forecast_canvas.create_text(
                x + card_width/2, y_offset + 60,
                text=f"{forecast.temperature:.0f}°",
                font=('Arial', 16, 'bold'), fill=self.colors['primary']
            )
            
            # Confidence bar
            conf_height = forecast.confidence * 100
            self.forecast_canvas.create_rectangle(
                x + 10, y_offset + 100,
                x + 20, y_offset + 100 + conf_height,
                fill=self.colors['success'], outline=''
            )
            
            # Precipitation chance
            if forecast.precipitation_chance > 0.1:
                self.forecast_canvas.create_text(
                    x + card_width/2, y_offset + 180,
                    text=f"{forecast.precipitation_chance:.0%}",
                    font=('Arial', 10), fill=self.colors['primary']
                )
        
        # Update scroll region
        self.forecast_canvas.configure(scrollregion=self.forecast_canvas.bbox('all'))
    
    def update_charts(self):
        """Update trend charts"""
        # Get historical data
        historical = self.data_manager.get_historical_data(48)
        
        if not historical.empty:
            # Temperature trend
            self.temp_ax.clear()
            self.temp_ax.plot(historical['timestamp'], historical['temperature'], 
                            'b-', linewidth=2, label='Actual')
            
            # Add forecast
            forecasts = self.prediction_engine.predict_ensemble(None, 24)
            forecast_times = [f.timestamp for f in forecasts]
            forecast_temps = [f.temperature for f in forecasts]
            
            self.temp_ax.plot(forecast_times, forecast_temps, 
                            'r--', linewidth=2, label='Forecast')
            
            # Confidence bands
            upper_bound = [f.temperature_high for f in forecasts]
            lower_bound = [f.temperature_low for f in forecasts]
            
            self.temp_ax.fill_between(forecast_times, lower_bound, upper_bound,
                                    alpha=0.2, color='red', label='Confidence')
            
            self.temp_ax.set_ylabel('Temperature (°F)')
            self.temp_ax.legend()
            self.temp_ax.grid(True, alpha=0.3)
            
            self.temp_canvas.draw()
            
            # Precipitation chart
            self.precip_ax.clear()
            precip_chances = [f.precipitation_chance for f in forecasts]
            
            self.precip_ax.bar(range(len(precip_chances)), 
                             [p * 100 for p in precip_chances],
                             color=self.colors['primary'], alpha=0.7)
            
            self.precip_ax.set_ylabel('Precipitation Chance (%)')
            self.precip_ax.set_xlabel('Hours Ahead')
            self.precip_ax.grid(True, alpha=0.3, axis='y')
            
            self.precip_canvas.draw()
    
    def manual_update(self):
        """Manual update triggered by user"""
        self.status_label.config(text="Updating...", foreground=self.colors['secondary'])
        self.update_current_weather()
        self.update_forecast_display()
        self.status_label.config(text="Updated", foreground=self.colors['success'])
        
    def toggle_auto_update(self):
        """Toggle automatic updates"""
        if self.auto_update_var.get():
            self.update_current_weather()
            
    def show_settings(self):
        """Show settings dialog"""
        SettingsDialog(self.root, self)

class SettingsDialog:
    """Settings dialog for the application"""
    
    def __init__(self, parent, app):
        self.app = app
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("400x300")
        
        # Make modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_settings_ui()
        
    def create_settings_ui(self):
        """Create settings interface"""
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # General settings
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        ttk.Label(general_frame, text="Update Interval (minutes):").grid(row=0, column=0, pady=10)
        self.interval_var = tk.IntVar(value=5)
        ttk.Spinbox(general_frame, from_=1, to=60, textvariable=self.interval_var,
                   width=10).grid(row=0, column=1)
        
        # Model settings
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Models")
        
        ttk.Label(model_frame, text="Ensemble Weights:").grid(row=0, column=0, pady=10)
        
        # Display settings
        display_frame = ttk.Frame(notebook)
        notebook.add(display_frame, text="Display")
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Button(button_frame, text="Save", command=self.save_settings).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side='right')
    
    def save_settings(self):
        """Save settings and close dialog"""
        # Apply settings to app
        # ...
        self.dialog.destroy()

@dataclass
class WeatherAlert:
    type: str
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    timestamp: datetime

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



# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherPredictorApp(root)
    root.mainloop()