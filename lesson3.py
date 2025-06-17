class PerformanceOptimizer:
    """Optimizes application performance"""
    
    def __init__(self, app):
        self.app = app
        self.cache = {}
        self.last_update = {}
        
    def cache_prediction(self, key, data, ttl=300):
        """Cache prediction data with time-to-live"""
        self.cache[key] = {
            'data': data,
            'expires': datetime.now() + timedelta(seconds=ttl)
        }
        
    def get_cached_prediction(self, key):
        """Get cached prediction if valid"""
        if key in self.cache:
            cached = self.cache[key]
            if datetime.now() < cached['expires']:
                return cached['data']
            else:
                del self.cache[key]
        return None
    
    def should_update(self, component, min_interval=60):
        """Check if component should update"""
        last = self.last_update.get(component, datetime.min)
        if (datetime.now() - last).seconds >= min_interval:
            self.last_update[component] = datetime.now()
            return True
        return False
    
    def optimize_chart_updates(self, chart_func):
        """Decorator to optimize chart updates"""
        def wrapper(*args, **kwargs):
            if self.should_update('chart', min_interval=30):
                return chart_func(*args, **kwargs)
        return wrapper
Smooth Animations and Transitions:
python
class AnimationManager:
    """Manages UI animations"""
    
    def __init__(self, root):
        self.root = root
        self.animations = []
        
    def fade_in(self, widget, duration=500, callback=None):
        """Fade in animation"""
        widget.configure(alpha=0)  # Custom attribute
        steps = 20
        delay = duration // steps
        
        def animate(step=0):
            if step <= steps:
                alpha = step / steps
                # Simulate fade with color interpolation
                widget.configure(alpha=alpha)
                self.root.after(delay, lambda: animate(step + 1))
            elif callback:
                callback()
                
        animate()
        
    def slide_in(self, widget, direction='left', duration=300):
        """Slide in animation"""
        # Get widget dimensions
        widget.update_idletasks()
        width = widget.winfo_width()
        height = widget.winfo_height()
        
        # Starting position
        if direction == 'left':
            start_x = -width
            end_x = 0
        elif direction == 'right':
            start_x = self.root.winfo_width()
            end_x = self.root.winfo_width() - width
            
        steps = 20
        delay = duration // steps
        
        def animate(step=0):
            if step <= steps:
                progress = step / steps
                # Ease-out curve
                progress = 1 - (1 - progress) ** 3
                
                x = start_x + (end_x - start_x) * progress
                widget.place(x=x, y=widget.winfo_y())
                
                self.root.after(delay, lambda: animate(step + 1))
                
        animate()
    
    def pulse(self, widget, duration=1000):
        """Pulse animation for alerts"""
        original_bg = widget.cget('background')
        
        def pulse_step(step=0):
            if step < 10:
                # Calculate intensity
                intensity = abs(5 - step) / 5
                # Interpolate color
                widget.configure(background=self._interpolate_color(
                    original_bg, '#ff0000', intensity
                ))
                self.root.after(duration // 10, lambda: pulse_step(step + 1))
            else:
                widget.configure(background=original_bg)
                
        pulse_step()
    
    def _interpolate_color(self, color1, color2, factor):
        """Interpolate between two colors"""
        # Simplified color interpolation
        return color1  # Placeholder

class ModernWeatherUI:
    """Modern UI enhancements"""
    
    def __init__(self, app):
        self.app = app
        self.setup_modern_ui()
        
    def setup_modern_ui(self):
        """Apply modern UI enhancements"""
        # Custom fonts
        self.fonts = {
            'heading': ('Segoe UI', 24, 'bold'),
            'subheading': ('Segoe UI', 16),
            'body': ('Segoe UI', 12),
            'small': ('Segoe UI', 10)
        }
        
        # Color scheme
        self.colors = {
            'bg_primary': '#1e1e1e',
            'bg_secondary': '#2d2d2d',
            'accent': '#0078d4',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'success': '#107c10',
            'warning': '#ff8c00',
            'error': '#d13438'
        }
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def apply_dark_theme(self):
        """Apply dark theme to application"""
        style = ttk.Style()
        
        # Configure dark theme
        style.theme_use('clam')
        
        style.configure('Dark.TFrame', background=self.colors['bg_primary'])
        style.configure('Dark.TLabel', 
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'])
        style.configure('Dark.TButton',
                       background=self.colors['accent'],
                       foreground=self.colors['text_primary'],
                       borderwidth=0,
                       focuscolor='none')
        
        # Hover effects
        style.map('Dark.TButton',
                 background=[('active', self.colors['accent']),
                           ('pressed', self.colors['bg_secondary'])])
    
    def create_modern_card(self, parent, title, content):
        """Create modern card widget"""
        card = tk.Frame(parent, bg=self.colors['bg_secondary'],
                       highlightbackground=self.colors['accent'],
                       highlightthickness=1)
        
        # Title
        title_label = tk.Label(card, text=title,
                             font=self.fonts['subheading'],
                             bg=self.colors['bg_secondary'],
                             fg=self.colors['text_primary'])
        title_label.pack(anchor='w', padx=20, pady=(15, 5))
        
        # Content
        content_frame = tk.Frame(card, bg=self.colors['bg_secondary'])
        content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        return content_frame
Error Handling and Recovery:
python
class ErrorHandler:
    """Comprehensive error handling"""
    
    def __init__(self, app):
        self.app = app
        self.error_log = []
        
    def handle_error(self, error, context="", severity="ERROR"):
        """Handle application errors"""
        # Log error
        error_entry = {
            'timestamp': datetime.now(),
            'error': str(error),
            'context': context,
            'severity': severity,
            'type': type(error).__name__
        }
        self.error_log.append(error_entry)
        
        # Determine action based on severity
        if severity == "CRITICAL":
            self.show_critical_error(error, context)
        elif severity == "ERROR":
            self.show_error_notification(error, context)
        else:  # WARNING or INFO
            self.log_warning(error, context)
            
    def show_critical_error(self, error, context):
        """Show critical error dialog"""
        message = f"A critical error occurred in {context}:\n\n{str(error)}\n\nThe application may need to restart."
        
        messagebox.showerror("Critical Error", message)
        
        # Attempt recovery
        self.attempt_recovery(context)
        
    def show_error_notification(self, error, context):
        """Show non-critical error notification"""
        # Create toast notification
        toast = tk.Toplevel(self.app.root)
        toast.wm_overrideredirect(True)
        
        # Position in bottom right
        toast.geometry("+{}+{}".format(
            self.app.root.winfo_x() + self.app.root.winfo_width() - 300,
            self.app.root.winfo_y() + self.app.root.winfo_height() - 100
        ))
        
        # Error message
        frame = tk.Frame(toast, bg='#d13438', padx=10, pady=10)
        frame.pack()
        
        tk.Label(frame, text=f"Error in {context}",
                fg='white', bg='#d13438',
                font=('Arial', 10, 'bold')).pack(anchor='w')
        
        tk.Label(frame, text=str(error)[:50] + "...",
                fg='white', bg='#d13438',
                font=('Arial', 9)).pack(anchor='w')
        
        # Auto-close after 5 seconds
        toast.after(5000, toast.destroy)
        
    def attempt_recovery(self, context):
        """Attempt to recover from critical error"""
        recovery_actions = {
            'data_fetch': self.recover_data_fetch,
            'prediction': self.recover_prediction,
            'ui_update': self.recover_ui_update
        }
        
        if context in recovery_actions:
            recovery_actions[context]()
        else:
            # Generic recovery
            self.app.reset_to_safe_state()
            
    def recover_data_fetch(self):
        """Recover from data fetch error"""
        # Use cached data
        self.app.use_cached_data = True
        self.app.status_label.config(text="Using cached data")
        
    def with_error_handling(self, context):
        """Decorator for error handling"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(e, context)
                    return None
            return wrapper
        return decorator
Final Complete Application:
python
class WeatherPredictorPro:
    """Production-ready weather predictor"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Weather Predictor Pro")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.setup_components()
        
        # Apply modern UI
        self.modern_ui = ModernWeatherUI(self)
        
        # Start application
        self.initialize()
        
    def setup_components(self):
        """Initialize all components"""
        self.data_manager = WeatherDataManager()
        self.prediction_engine = PredictionEngine(self.load_models())
        self.alert_system = WeatherAlertSystem()
        self.location_manager = LocationManager()
        self.export_manager = ExportManager(self)
        self.error_handler = ErrorHandler(self)
        self.performance = PerformanceOptimizer(self)
        self.animator = AnimationManager(self.root)
        
    def load_models(self):
        """Load all ML models"""
        models = {}
        
        # Load with error handling
        for model_name in ['linear_regression', 'random_forest', 'arima']:
            try:
                model_path = f'models/{model_name}_model.pkl'
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                
        return models
    
    def initialize(self):
        """Initialize application"""
        # Setup UI
        self.setup_ui()
        
        # Load initial data
        self.load_initial_data()
        
        # Start update cycles
        self.start_update_cycles()
        
        # Show welcome
        self.show_welcome()
        
    def setup_ui(self):
        """Create complete UI"""
        # Main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        
        # Dashboard tab
        self.dashboard = self.create_dashboard()
        self.notebook.add(self.dashboard, text="Dashboard")
        
        # Detailed forecast tab
        self.forecast_tab = self.create_forecast_tab()
        self.notebook.add(self.forecast_tab, text="Detailed Forecast")
        
        # Maps tab
        self.maps_tab = self.create_maps_tab()
        self.notebook.add(self.maps_tab, text="Weather Maps")
        
        # Alerts tab
        self.alerts_tab = self.create_alerts_tab()
        self.notebook.add(self.alerts_tab, text="Alerts")
        
        # Settings tab
        self.settings_tab = self.create_settings_tab()
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Status bar
        self.create_status_bar()
        
    def create_dashboard(self):
        """Create main dashboard"""
        dashboard = ttk.Frame(self.notebook)
        
        # Current weather card
        current_frame = self.modern_ui.create_modern_card(
            dashboard, "Current Weather", None
        )
        current_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Quick forecast
        forecast_frame = self.modern_ui.create_modern_card(
            dashboard, "Quick Forecast", None
        )
        forecast_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        return dashboard
    
    def show_welcome(self):
        """Show welcome message"""
        welcome = tk.Toplevel(self.root)
        welcome.title("Welcome")
        welcome.geometry("400x200")
        
        ttk.Label(welcome, text="Welcome to Weather Predictor Pro!",
                 font=('Arial', 16, 'bold')).pack(pady=20)
        
        ttk.Label(welcome, text="Your advanced weather forecasting companion",
                 font=('Arial', 12)).pack()
        
        ttk.Button(welcome, text="Get Started",
                  command=welcome.destroy).pack(pady=20)
        
        # Center on screen
        welcome.transient(self.root)
        welcome.grab_set()
        
    def run(self):
        """Run the application"""
        self.root.mainloop()

# Create and run the application
if __name__ == "__main__":
    app = WeatherPredictorPro()
    app.run()
Best Practices Summary:
python
class WeatherAppBestPractices:
    """Summary of best practices for weather applications"""
    
    @staticmethod
    def ui_design():
        return {
            "Clarity": "Show most important info prominently",
            "Responsiveness": "Update smoothly without freezing",
            "Accessibility": "Use clear fonts and good contrast",
            "Consistency": "Maintain consistent design language",
            "Feedback": "Always show system status"
        }
    
    @staticmethod
    def data_handling():
        return {
            "Caching": "Cache data to reduce API calls",
            "Validation": "Validate all data before display",
            "Error Handling": "Gracefully handle missing data",
            "Updates": "Balance freshness with performance",
            "Storage": "Store historical data efficiently"
        }
    
    @staticmethod
    def predictions():
        return {
            "Uncertainty": "Always show confidence levels",
            "Multiple Models": "Use ensemble for better accuracy",
            "Validation": "Continuously validate predictions",
            "Explanation": "Help users understand predictions",
            "Limitations": "Be clear about forecast limits"
        }
