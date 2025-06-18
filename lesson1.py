import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import threading
import queue

@dataclass
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
        # Simulate API call
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
        
        # Simulate historical data
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
    
    def __init__(self, models: Dict):
        self.models = models
        self.ensemble_weights = {'linear': 0.3, 'rf': 0.5, 'arima': 0.2}
        
    def predict_ensemble(self, features: pd.DataFrame, horizon: int) -> List[Forecast]:
        """Make ensemble predictions"""
        predictions = []
        
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
        
        # Setup UI
        self.setup_ui()
        
        # Start data updates
        self.update_current_weather()
        
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
        
    def create_charts_panel(self, parent):
        """Create charts section"""
        charts_frame = ttk.LabelFrame(parent, text="Trends", padding="10")
        charts_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                         pady=(10, 0))
        
        # Chart notebook
        self.chart_notebook = ttk.Notebook(charts_frame)
        self.chart_notebook.pack(fill='both', expand=True)
        
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

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherPredictorApp(root)
    root.mainloop()