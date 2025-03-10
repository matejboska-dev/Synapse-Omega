import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import os
import matplotlib.dates as mdates
import logging
from tqdm import tqdm
import threading
import queue
from PIL import Image, ImageTk
import io
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.ticker as mtick

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("interactive-news-visualization")

# Default database configuration
DEFAULT_DB_SERVER = "193.85.203.188"
DEFAULT_DB_NAME = "boska"
DEFAULT_DB_USER = "boska"
DEFAULT_DB_PASSWORD = "123456"

# Create output directory for graphs
def create_output_dir():
    output_dir = "visualization_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Download NLTK resources if needed
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

# Function to clean and preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Main Application Class
class NewsVisualizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synapse - Vizualizace dat")  # Shorter title
        self.geometry("900x600")  # Smaller default size
        self.configure(bg="#f0f0f0")
        self.minsize(800, 500)  # Smaller minimum size
        
        # Initialize variables
        self.df = None
        self.conn = None
        self.output_dir = create_output_dir()
        self.queue = queue.Queue()
        
        # Download NLTK resources
        threading.Thread(target=download_nltk_resources).start()
        
        # Create styles
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use 'clam' theme as base
        
        # Configure colors
        self.bg_color = "#f0f0f0"
        self.fg_color = "#333333"
        self.accent_color = "#4a6ea9"
        self.success_color = "#28a745"
        self.warning_color = "#ffc107"
        self.error_color = "#dc3545"
        
        # Configure styles with smaller fonts
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 9))
        self.style.configure("TButton", background=self.accent_color, foreground="white", font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 12, "bold"))
        self.style.configure("Subtitle.TLabel", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10, "bold"))
        self.style.configure("Status.TLabel", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 8))
        
        # Build UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Connection frame
        self.connection_frame = ttk.LabelFrame(self.main_frame, text="Připojení k databázi")
        self.connection_frame.pack(fill=tk.X, padx=2, pady=2)
        
        # Connection form - more compact
        form_frame = ttk.Frame(self.connection_frame)
        form_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Server
        ttk.Label(form_frame, text="Server:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.server_var = tk.StringVar(value=DEFAULT_DB_SERVER)
        ttk.Entry(form_frame, textvariable=self.server_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        
        # Database
        ttk.Label(form_frame, text="Databáze:").grid(row=0, column=2, sticky=tk.W, padx=2, pady=2)
        self.db_var = tk.StringVar(value=DEFAULT_DB_NAME)
        ttk.Entry(form_frame, textvariable=self.db_var, width=20).grid(row=0, column=3, sticky=tk.W, padx=2, pady=2)
        
        # Username
        ttk.Label(form_frame, text="Uživatel:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.user_var = tk.StringVar(value=DEFAULT_DB_USER)
        ttk.Entry(form_frame, textvariable=self.user_var, width=20).grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        
        # Password
        ttk.Label(form_frame, text="Heslo:").grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)
        self.password_var = tk.StringVar(value=DEFAULT_DB_PASSWORD)
        ttk.Entry(form_frame, textvariable=self.password_var, width=20, show="*").grid(row=1, column=3, sticky=tk.W, padx=2, pady=2)
        
        # Connect button
        button_frame = ttk.Frame(self.connection_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.connect_button = ttk.Button(button_frame, text="Připojit k databázi", command=self.connect_to_db)
        self.connect_button.pack(side=tk.RIGHT, padx=2, pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Čekám na připojení k databázi...")
        self.status_label = ttk.Label(button_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Visualization frame (initially hidden)
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Vizualizace dat")
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Dashboard tab
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        
        # Setup dashboard
        self.setup_dashboard()
        
        # Status bar
        self.status_bar = ttk.Frame(self)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_text = tk.StringVar(value="Připraven")
        ttk.Label(self.status_bar, textvariable=self.status_text, style="Status.TLabel").pack(side=tk.LEFT, padx=5, pady=1)
        
        # Version info
        ttk.Label(self.status_bar, text="Synapse v1.0", style="Status.TLabel").pack(side=tk.RIGHT, padx=5, pady=1)
    
    def setup_dashboard(self):
        # Přidat canvas s scrollbarem pro celý obsah
        canvas = tk.Canvas(self.dashboard_tab)
        scrollbar = ttk.Scrollbar(self.dashboard_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Left panel - visualization options
        options_frame = ttk.LabelFrame(scrollable_frame, text="Vizualizace")
        options_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        # Compact checkboxes layout
        self.viz_options = {
            "articles_by_source": tk.BooleanVar(value=True),
            "articles_by_date": tk.BooleanVar(value=True),
            "article_length_distribution": tk.BooleanVar(value=True),
            "word_count_distribution": tk.BooleanVar(value=True),
            "articles_by_category": tk.BooleanVar(value=True),
            "length_word_correlation": tk.BooleanVar(value=True),
            "articles_source_time": tk.BooleanVar(value=True),
            "word_cloud": tk.BooleanVar(value=True),
            "avg_length_by_source": tk.BooleanVar(value=True),
            "top_words": tk.BooleanVar(value=True),
            "publication_time_of_day": tk.BooleanVar(value=True),
            "length_by_source_boxplot": tk.BooleanVar(value=True),
            "articles_by_day_of_week": tk.BooleanVar(value=True),
            "avg_word_length": tk.BooleanVar(value=True),
        }
        
        # Visualization names - shortened
        viz_names = {
            "articles_by_source": "Články dle zdroje",
            "articles_by_date": "Články dle data",
            "article_length_distribution": "Distribuce délky",
            "word_count_distribution": "Distribuce slov",
            "articles_by_category": "Dle kategorie",
            "length_word_correlation": "Korelace délky/slov",
            "articles_source_time": "Zdroj a čas",
            "word_cloud": "Word Cloud",
            "avg_length_by_source": "Prům. délka dle zdroje",
            "top_words": "Nejčastější slova",
            "publication_time_of_day": "Čas publikace",
            "length_by_source_boxplot": "Box plot délky",
            "articles_by_day_of_week": "Den v týdnu",
            "avg_word_length": "Prům. délka slov",
        }
        
        # Create checkboxes in a more compact grid (2 columns)
        row, col = 0, 0
        for key, var in self.viz_options.items():
            ttk.Checkbutton(options_frame, text=viz_names[key], variable=var).grid(
                row=row, column=col, sticky=tk.W, padx=2, pady=1
            )
            col += 1
            if col > 1:  # 2 columns
                col = 0
                row += 1
        
        # Button panel - make more compact
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=5)
        
        ttk.Button(button_frame, text="Vše", command=self.select_all).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="Nic", command=self.deselect_all).grid(row=0, column=1, padx=2)
        
        # Generate and save buttons
        self.generate_button = ttk.Button(button_frame, text="Generovat", command=self.start_visualization)
        self.generate_button.grid(row=0, column=2, padx=2)
        
        self.save_button = ttk.Button(button_frame, text="Uložit", command=self.save_all_visualizations)
        self.save_button.grid(row=0, column=3, padx=2)
        
        # Info panel - more compact
        info_frame = ttk.LabelFrame(scrollable_frame, text="Info")
        info_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        # Data info in a compact format
        self.data_info = {
            "total_articles": tk.StringVar(value="0"),
            "unique_sources": tk.StringVar(value="0"),
            "date_range": tk.StringVar(value="N/A"),
            "avg_article_length": tk.StringVar(value="0"),
            "avg_word_count": tk.StringVar(value="0"),
        }
        
        info_labels = {
            "total_articles": "Článků:",
            "unique_sources": "Zdrojů:",
            "date_range": "Období:",
            "avg_article_length": "Prům. délka:",
            "avg_word_count": "Prům. slov:",
        }
        
        # Compact grid layout for info
        row = 0
        for key, label_text in info_labels.items():
            ttk.Label(info_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=2, pady=1)
            ttk.Label(info_frame, textvariable=self.data_info[key]).grid(row=row, column=1, sticky=tk.W, padx=2, pady=1)
            row += 1
        
        # Preview frame
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Náhled")
        preview_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=2, pady=5)
        
        # Preview message
        self.preview_msg = ttk.Label(preview_frame, text="Připojte se k databázi a vygenerujte vizualizace.")
        self.preview_msg.pack(expand=True, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(scrollable_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        # Configure grid weights
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(2, weight=1)
    
    def select_all(self):
        for var in self.viz_options.values():
            var.set(True)
    
    def deselect_all(self):
        for var in self.viz_options.values():
            var.set(False)
    
    def connect_to_db(self):
        # Get connection parameters
        server = self.server_var.get()
        database = self.db_var.get()
        username = self.user_var.get()
        password = self.password_var.get()
        
        # Update status
        self.status_var.set("Připojuji k databázi...")
        self.status_text.set("Připojování...")
        self.update_idletasks()
        
        # Try to connect
        try:
            conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
            self.conn = pyodbc.connect(conn_str)
            
            # Fetch data
            self.status_var.set("Načítám data...")
            self.update_idletasks()
            
            query = """
            SELECT SourceName, Title, ArticleUrl, PublicationDate, Category, 
                   ArticleLength, WordCount, ArticleText
            FROM Articles
            """
            self.df = pd.read_sql(query, self.conn)
            
            # Show visualization frame
            self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            
            # Update status
            self.status_var.set(f"Připojeno. Načteno {len(self.df)} článků.")
            self.status_text.set("Připraven")
            
            # Update data info
            self.update_data_info()
            
            # Disable connect button
            self.connect_button.config(state="disabled")
            
            # Update preview message
            if self.preview_msg.winfo_exists():
                self.preview_msg.config(text="Klikněte na 'Generovat' pro zobrazení grafů.")
            
            messagebox.showinfo("Úspěch", f"Úspěšně připojeno. Načteno {len(self.df)} článků.")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Chyba při připojování k databázi: {error_msg}")
            self.status_var.set(f"Chyba: {error_msg[:30]}...")
            self.status_text.set("Chyba připojení")
            messagebox.showerror("Chyba připojení", f"Nepodařilo se připojit k databázi:\n{error_msg}")
    
    def update_data_info(self):
        if self.df is not None:
            # Total articles
            self.data_info["total_articles"].set(str(len(self.df)))
            
            # Unique sources
            self.data_info["unique_sources"].set(str(self.df['SourceName'].nunique()))
            
            # Date range
            try:
                self.df['PublicationDate'] = pd.to_datetime(self.df['PublicationDate'], errors='coerce')
                min_date = self.df['PublicationDate'].min().strftime('%d.%m.%Y')
                max_date = self.df['PublicationDate'].max().strftime('%d.%m.%Y')
                self.data_info["date_range"].set(f"{min_date} - {max_date}")
            except:
                self.data_info["date_range"].set("N/A")
            
            # Average article length
            self.data_info["avg_article_length"].set(f"{self.df['ArticleLength'].mean():.1f}")
            
            # Average word count
            self.data_info["avg_word_count"].set(f"{self.df['WordCount'].mean():.1f}")
    
    def start_visualization(self):
        if self.df is None:
            messagebox.showwarning("Upozornění", "Nejprve se připojte k databázi.")
            return
        
        # Check if any visualization is selected
        if not any(var.get() for var in self.viz_options.values()):
            messagebox.showwarning("Upozornění", "Vyberte alespoň jednu vizualizaci.")
            return
        
        # Disable generate button
        self.generate_button.config(state="disabled")
        self.save_button.config(state="disabled")
        
        # Start visualization thread
        threading.Thread(target=self.generate_visualizations).start()
        
        # Start checking queue
        self.after(100, self.check_queue)
    
    def generate_visualizations(self):
        try:
            # Get selected visualizations
            selected = [key for key, var in self.viz_options.items() if var.get()]
            total = len(selected)
            
            # Create tabs for selected visualizations
            self.queue.put(("create_tabs", selected))
            
            # Generate each visualization
            for i, viz_key in enumerate(selected):
                # Update progress
                progress = (i / total) * 100
                self.queue.put(("update_progress", progress))
                self.queue.put(("update_status", f"Generuji: {viz_key}..."))
                
                # Generate visualization
                fig = self.create_visualization(viz_key)
                
                # Add to tab
                self.queue.put(("add_to_tab", (viz_key, fig)))
            
            # Final update
            self.queue.put(("update_progress", 100))
            self.queue.put(("update_status", "Vizualizace dokončeny"))
            self.queue.put(("enable_buttons", None))
            
        except Exception as e:
            logger.error(f"Chyba při generování vizualizací: {e}")
            self.queue.put(("error", str(e)))
    
    def check_queue(self):
        try:
            while True:
                action, data = self.queue.get_nowait()
                
                if action == "update_progress":
                    self.progress_var.set(data)
                
                elif action == "update_status":
                    self.status_text.set(data)
                
                elif action == "error":
                    messagebox.showerror("Chyba", f"Chyba při generování vizualizací:\n{data}")
                    self.generate_button.config(state="normal")
                    self.save_button.config(state="normal")
                
                elif action == "create_tabs":
                    # Clear existing tabs except dashboard
                    for tab in self.notebook.tabs():
                        if self.notebook.tab(tab, "text") != "Dashboard":
                            self.notebook.forget(tab)
                    
                    # Create new tabs
                    for viz_key in data:
                        frame = ttk.Frame(self.notebook)
                        self.notebook.add(frame, text=self.get_viz_title(viz_key))
                
                elif action == "add_to_tab":
                    viz_key, fig = data
                    
                    # Find tab index
                    viz_title = self.get_viz_title(viz_key)
                    tab_idx = None
                    
                    for i, item in enumerate(self.notebook.tabs()):
                        if self.notebook.tab(item, "text") == viz_title:
                            tab_idx = i
                            break
                    
                    if tab_idx is not None:
                        # Get tab frame
                        frame = self.notebook.children[self.notebook.tabs()[tab_idx].split(".")[-1]]
                        
                        # Clear frame
                        for widget in frame.winfo_children():
                            widget.destroy()
                        
                        # Add figure to frame
                        canvas = FigureCanvasTkAgg(fig, master=frame)
                        canvas.draw()
                        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                        
                        # Add toolbar
                        toolbar = NavigationToolbar2Tk(canvas, frame)
                        toolbar.update()
                        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                        
                        # Update preview if this is the first visualization
                        if self.preview_msg.winfo_exists():
                            self.preview_msg.destroy()
                            
                            # Create preview canvas in preview frame
                            preview_frame = self.preview_msg.master
                            preview_fig = self.create_visualization(viz_key, preview=True)
                            preview_canvas = FigureCanvasTkAgg(preview_fig, master=preview_frame)
                            preview_canvas.draw()
                            preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                elif action == "enable_buttons":
                    self.generate_button.config(state="normal")
                    self.save_button.config(state="normal")
                
                elif action == "show_info":
                    messagebox.showinfo("Informace", data)
                
                self.queue.task_done()
                
        except queue.Empty:
            # Schedule next check
            self.after(100, self.check_queue)
    
    def get_viz_title(self, viz_key):
        # Map visualization keys to human-readable titles
        titles = {
            "articles_by_source": "Články dle zdroje",
            "articles_by_date": "Články dle data",
            "article_length_distribution": "Distribuce délky",
            "word_count_distribution": "Distribuce slov",
            "articles_by_category": "Dle kategorie",
            "length_word_correlation": "Korelace délky/slov",
            "articles_source_time": "Zdroj a čas",
            "word_cloud": "Word Cloud",
            "avg_length_by_source": "Prům. délka dle zdroje",
            "top_words": "Nejčastější slova",
            "publication_time_of_day": "Čas publikace",
            "length_by_source_boxplot": "Box plot délky",
            "articles_by_day_of_week": "Den v týdnu",
            "avg_word_length": "Prům. délka slov",
        }
        return titles.get(viz_key, viz_key)
    
    def create_visualization(self, viz_key, preview=False):
        # Set figure size based on whether this is a preview
        figsize = (6, 4) if preview else (8, 6)
        
        try:
            # Create appropriate visualization based on key
            if viz_key == "articles_by_source":
                return self.plot_articles_by_source(figsize)
            elif viz_key == "articles_by_date":
                return self.plot_articles_by_date(figsize)
            elif viz_key == "article_length_distribution":
                return self.plot_article_length_distribution(figsize)
            elif viz_key == "word_count_distribution":
                return self.plot_word_count_distribution(figsize)
            elif viz_key == "articles_by_category":
                return self.plot_articles_by_category(figsize)
            elif viz_key == "length_word_correlation":
                return self.plot_length_word_correlation(figsize)
            elif viz_key == "articles_source_time":
                return self.plot_articles_source_time(figsize)
            elif viz_key == "word_cloud":
                return self.create_word_cloud(figsize)
            elif viz_key == "avg_length_by_source":
                return self.plot_avg_length_by_source(figsize)
            elif viz_key == "top_words":
                return self.plot_top_words(figsize)
            elif viz_key == "publication_time_of_day":
                return self.plot_publication_time_of_day(figsize)
            elif viz_key == "length_by_source_boxplot":
                return self.plot_length_by_source_boxplot(figsize)
            elif viz_key == "articles_by_day_of_week":
                return self.plot_articles_by_day_of_week(figsize)
            elif viz_key == "avg_word_length":
                return self.plot_avg_word_length(figsize)
            else:
                # Default empty figure
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f"Neznámá vizualizace: {viz_key}", ha='center', va='center', fontsize=14)
                return fig
        except Exception as e:
            logger.error(f"Chyba při generování vizualizace {viz_key}: {e}")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Chyba: {str(e)}", ha='center', va='center', fontsize=14)
            return fig
    
    def plot_articles_by_source(self, figsize):
        plt.figure(figsize=figsize)
        source_counts = self.df['SourceName'].value_counts()
        
        # Plot horizontal bar chart
        ax = sns.barplot(x=source_counts.values, y=source_counts.index, palette='viridis')
        
        # Add labels and title
        plt.title('Počet článků podle zdroje', fontsize=12)
        plt.xlabel('Počet článků', fontsize=9)
        plt.ylabel('Zdroj', fontsize=9)
        
        # Add count labels to the bars
        for i, v in enumerate(source_counts.values):
            ax.text(v + 3, i, str(v), va='center', fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_articles_by_date(self, figsize):
        # Convert PublicationDate to datetime
        self.df['PublicationDate'] = pd.to_datetime(self.df['PublicationDate'], errors='coerce')
        
        # Drop rows with missing publication dates
        df_date = self.df.dropna(subset=['PublicationDate'])
        
        # Group by date
        date_counts = df_date.groupby(df_date['PublicationDate'].dt.date).size().reset_index(name='count')
        date_counts['PublicationDate'] = pd.to_datetime(date_counts['PublicationDate'])
        date_counts = date_counts.sort_values('PublicationDate')
        
        plt.figure(figsize=figsize)
        plt.plot(date_counts['PublicationDate'], date_counts['count'], marker='o', linestyle='-', color='teal')
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show date every week
        plt.gcf().autofmt_xdate()  # Rotate date labels
        
        plt.title('Počet článků podle data publikace', fontsize=12)
        plt.xlabel('Datum publikace', fontsize=9)
        plt.ylabel('Počet článků', fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_article_length_distribution(self, figsize):
        plt.figure(figsize=figsize)
        
        # Plot histogram
        sns.histplot(self.df['ArticleLength'], bins=30, kde=True, color='purple')
        
        plt.title('Distribuce délky článků', fontsize=12)
        plt.xlabel('Délka článku (znaky)', fontsize=9)
        plt.ylabel('Frekvence', fontsize=9)
        
        # Add median and mean lines
        median_length = self.df['ArticleLength'].median()
        mean_length = self.df['ArticleLength'].mean()
        
        plt.axvline(median_length, color='red', linestyle='--', label=f'Med: {median_length:.0f}')
        plt.axvline(mean_length, color='green', linestyle='--', label=f'Avg: {mean_length:.0f}')
        plt.legend(fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_word_count_distribution(self, figsize):
        plt.figure(figsize=figsize)
        
        # Plot histogram
        sns.histplot(self.df['WordCount'], bins=30, kde=True, color='blue')
        
        plt.title('Distribuce počtu slov', fontsize=12)
        plt.xlabel('Počet slov', fontsize=9)
        plt.ylabel('Frekvence', fontsize=9)
        
        # Add median and mean lines
        median_count = self.df['WordCount'].median()
        mean_count = self.df['WordCount'].mean()
        
        plt.axvline(median_count, color='red', linestyle='--', label=f'Med: {median_count:.0f}')
        plt.axvline(mean_count, color='green', linestyle='--', label=f'Avg: {mean_count:.0f}')
        plt.legend(fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_articles_by_category(self, figsize):
        # Fill NaN values
        df_copy = self.df.copy()
        df_copy['Category'] = df_copy['Category'].fillna('Nezařazeno')
        
        # If category is empty string, replace with 'Nezařazeno'
        df_copy.loc[df_copy['Category'] == '', 'Category'] = 'Nezařazeno'
        
        # Get the top 10 categories (reduced from 15 for smaller screens)
        category_counts = df_copy['Category'].value_counts().head(10)
        
        plt.figure(figsize=figsize)
        
        # Plot horizontal bar chart
        ax = sns.barplot(x=category_counts.values, y=category_counts.index, palette='magma')
        
        # Add labels and title
        plt.title('Počet článků podle kategorie (Top 10)', fontsize=12)
        plt.xlabel('Počet článků', fontsize=9)
        plt.ylabel('Kategorie', fontsize=9)
        
        # Add count labels to the bars
        for i, v in enumerate(category_counts.values):
            ax.text(v + 3, i, str(v), va='center', fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_length_word_correlation(self, figsize):
        plt.figure(figsize=figsize)
        
        # Create scatter plot with regression line
        sns.regplot(x='ArticleLength', y='WordCount', data=self.df, scatter_kws={'alpha':0.5, 's':10}, line_kws={'color':'red'})
        
        plt.title('Korelace: Délka článku vs. Počet slov', fontsize=12)
        plt.xlabel('Délka článku (znaky)', fontsize=9)
        plt.ylabel('Počet slov', fontsize=9)
        
        # Calculate and display correlation coefficient
        correlation = self.df['ArticleLength'].corr(self.df['WordCount'])
        plt.annotate(f'Korelace: {correlation:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_articles_source_time(self, figsize):
        # Convert PublicationDate to datetime
        df_copy = self.df.copy()
        df_copy['PublicationDate'] = pd.to_datetime(df_copy['PublicationDate'], errors='coerce')
        
        # Drop rows with missing publication dates
        df_date = df_copy.dropna(subset=['PublicationDate'])
        
        # Add month-year column for grouping
        df_date['MonthYear'] = df_date['PublicationDate'].dt.to_period('M')
        
        # Get the top 5 sources (reduced from 8 for smaller screens)
        top_sources = df_date['SourceName'].value_counts().head(5).index.tolist()
        df_top = df_date[df_date['SourceName'].isin(top_sources)]
        
        # Group by source and month
        source_month = df_top.groupby(['MonthYear', 'SourceName']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=figsize)
        source_month.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        plt.title('Počet článků podle zdroje a měsíce', fontsize=12)
        plt.xlabel('Měsíc', fontsize=9)
        plt.ylabel('Počet článků', fontsize=9)
        plt.legend(title='Zdroj', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def create_word_cloud(self, figsize):
        # Combine all article texts
        all_text = ' '.join(self.df['ArticleText'].dropna().astype(str).tolist())
        
        # Preprocess text
        all_text = preprocess_text(all_text)
        
        # Get Czech stopwords
        czech_stopwords = set(stopwords.words('czech'))
        
        # Add custom Czech stopwords often found in news
        custom_stopwords = {
            'podle', 'proto', 'nové', 'jeho', 'které', 'také', 
            'jsme', 'mezi', 'může', 'řekl', 'uvedl', 'další',
            'této', 'byly', 'bude', 'byla', 'jako', 'více', 'však',
            'řekla', 'této', 'roku', 'letech', 'korun', 'lidí', 'dnes',
            'měl', 'stále', 'mimo', 'nový', 'době', 'již', 'svého', 'tato',
            'prý', 'jde', 'uvádí', 'kdy', 'není', 'české', 'včera', 'tak',
            'mají', 'informoval', 'sdělil', 'zatím', 'bude', 'lidé', 'což',
            'kterou', 'něco', 'toho', 'stát', 'tisíc', 'kterou', 'tím', 
            'právě', 'nejen', 'této', 'tento', 'toto'
        }
        czech_stopwords.update(custom_stopwords)
        
        # Create and generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=czech_stopwords, 
            max_words=100,  # Reduced from 200 for smaller screens
            collocations=False
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud nejčastějších slov v článcích', fontsize=12)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_avg_length_by_source(self, figsize):
        # Calculate average article length by source
        avg_length = self.df.groupby('SourceName')['ArticleLength'].mean().sort_values(ascending=False)
        
        plt.figure(figsize=figsize)
        
        # Plot horizontal bar chart
        ax = sns.barplot(x=avg_length.values, y=avg_length.index, palette='plasma')
        
        # Add labels and title
        plt.title('Průměrná délka článku podle zdroje', fontsize=12)
        plt.xlabel('Průměrná délka (znaky)', fontsize=9)
        plt.ylabel('Zdroj', fontsize=9)
        
        # Add values to the bars
        for i, v in enumerate(avg_length.values):
            ax.text(v + 3, i, f"{v:.0f}", va='center', fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_top_words(self, figsize):
        # Combine all article texts
        all_text = ' '.join(self.df['ArticleText'].dropna().astype(str).tolist())
        
        # Preprocess text
        all_text = preprocess_text(all_text)
        
        # Get Czech stopwords
        czech_stopwords = set(stopwords.words('czech'))
        
        # Add custom Czech stopwords often found in news
        custom_stopwords = {
            'podle', 'proto', 'nové', 'jeho', 'které', 'také', 
            'jsme', 'mezi', 'může', 'řekl', 'uvedl', 'další',
            'této', 'byly', 'bude', 'byla', 'jako', 'více', 'však',
            'řekla', 'této', 'roku', 'letech', 'korun', 'lidí', 'dnes',
            'měl', 'stále', 'mimo', 'nový', 'době', 'již', 'svého', 'tato',
            'prý', 'jde', 'uvádí', 'kdy', 'není', 'české', 'včera', 'tak',
            'mají', 'informoval', 'sdělil', 'zatím', 'bude', 'lidé', 'což',
            'kterou', 'něco', 'toho', 'stát', 'tisíc'
        }
        czech_stopwords.update(custom_stopwords)
        
        # Tokenize text
        words = nltk.word_tokenize(all_text)
        
        # Remove stopwords and short words
        words = [word for word in words if word not in czech_stopwords and len(word) > 2]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Get top 15 words (reduced from 20 for smaller screens)
        top_words = dict(word_freq.most_common(15))
        
        plt.figure(figsize=figsize)
        
        # Plot horizontal bar chart
        plt.barh(list(top_words.keys())[::-1], list(top_words.values())[::-1], color='skyblue')
        
        plt.title('Top 15 nejčastějších slov v článcích', fontsize=12)
        plt.xlabel('Frekvence', fontsize=9)
        plt.ylabel('Slovo', fontsize=9)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_publication_time_of_day(self, figsize):
        # Convert PublicationDate to datetime
        df_copy = self.df.copy()
        df_copy['PublicationDate'] = pd.to_datetime(df_copy['PublicationDate'], errors='coerce')
        
        # Drop rows with missing publication dates
        df_time = df_copy.dropna(subset=['PublicationDate'])
        
        # Extract hour of publication
        df_time['Hour'] = df_time['PublicationDate'].dt.hour
        
        # Count articles by hour
        hour_counts = df_time.groupby('Hour').size()
        
        # Plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=hour_counts.index, y=hour_counts.values, palette='crest')
        
        plt.title('Počet článků podle hodiny publikace', fontsize=12)
        plt.xlabel('Hodina (0-23)', fontsize=9)
        plt.ylabel('Počet článků', fontsize=9)
        plt.xticks(range(0, 24, 2))  # Show every other hour to save space
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels to the bars (only for bars with significant values)
        for i, v in enumerate(hour_counts.values):
            if v > hour_counts.values.max() * 0.2:  # Only label significant bars
                ax.text(i, v + 1, str(v), ha='center', fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_length_by_source_boxplot(self, figsize):
        # Get top 7 sources by number of articles (reduced from 10 for smaller screens)
        top_sources = self.df['SourceName'].value_counts().head(7).index.tolist()
        df_top = self.df[self.df['SourceName'].isin(top_sources)]
        
        plt.figure(figsize=figsize)
        ax = sns.boxplot(x='SourceName', y='ArticleLength', data=df_top, palette='Set3')
        
        plt.title('Distribuce délky článku podle zdroje', fontsize=12)
        plt.xlabel('Zdroj', fontsize=9)
        plt.ylabel('Délka článku (znaky)', fontsize=9)
        plt.xticks(rotation=45, fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_articles_by_day_of_week(self, figsize):
        # Convert PublicationDate to datetime
        df_copy = self.df.copy()
        df_copy['PublicationDate'] = pd.to_datetime(df_copy['PublicationDate'], errors='coerce')
        
        # Drop rows with missing publication dates
        df_date = df_copy.dropna(subset=['PublicationDate'])
        
        # Create day of week column (0 = Monday, 6 = Sunday)
        df_date['DayOfWeek'] = df_date['PublicationDate'].dt.dayofweek
        
        # Map numeric day to day name
        day_names = {0: 'Po', 1: 'Út', 2: 'St', 3: 'Čt', 
                    4: 'Pá', 5: 'So', 6: 'Ne'}  # Shortened names
        df_date['DayName'] = df_date['DayOfWeek'].map(day_names)
        
        # Group by day of week
        day_counts = df_date.groupby('DayName').size()
        
        # Get correct order of days
        ordered_days = [day_names[i] for i in range(7)]
        day_counts = day_counts.reindex(ordered_days)
        
        plt.figure(figsize=figsize)
        ax = sns.barplot(x=day_counts.index, y=day_counts.values, palette='viridis')
        
        plt.title('Počet článků podle dne v týdnu', fontsize=12)
        plt.xlabel('Den v týdnu', fontsize=9)
        plt.ylabel('Počet článků', fontsize=9)
        
        # Add count labels to the bars
        for i, v in enumerate(day_counts.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_avg_word_length(self, figsize):
        # Calculate average word length for each article
        def calc_avg_word_len(text):
            if not isinstance(text, str) or not text:
                return 0
            words = text.split()
            if not words:
                return 0
            return sum(len(word) for word in words) / len(words)
        
        df_copy = self.df.copy()
        df_copy['AvgWordLength'] = df_copy['ArticleText'].apply(calc_avg_word_len)
        
        plt.figure(figsize=figsize)
        
        # Plot histogram
        sns.histplot(df_copy['AvgWordLength'].dropna(), bins=20, kde=True, color='orange')
        
        plt.title('Průměrná délka slov v článcích', fontsize=12)
        plt.xlabel('Průměrná délka slova (znaky)', fontsize=9)
        plt.ylabel('Frekvence', fontsize=9)
        
        # Add median and mean lines
        median_len = df_copy['AvgWordLength'].median()
        mean_len = df_copy['AvgWordLength'].mean()
        
        plt.axvline(median_len, color='red', linestyle='--', label=f'Med: {median_len:.2f}')
        plt.axvline(mean_len, color='green', linestyle='--', label=f'Avg: {mean_len:.2f}')
        plt.legend(fontsize=8)
        
        plt.tight_layout()
        return plt.gcf()
    
    def save_all_visualizations(self):
        if self.df is None:
            messagebox.showwarning("Upozornění", "Nejprve se připojte k databázi.")
            return
        
        # Ask for directory
        output_dir = filedialog.askdirectory(title="Vyberte složku pro uložení grafů")
        if not output_dir:
            return
        
        # Disable buttons
        self.generate_button.config(state="disabled")
        self.save_button.config(state="disabled")
        
        # Start saving thread
        threading.Thread(target=self.save_visualizations_thread, args=(output_dir,)).start()
    
    def save_visualizations_thread(self, output_dir):
        try:
            # Get selected visualizations
            selected = [key for key, var in self.viz_options.items() if var.get()]
            if not selected:
                selected = list(self.viz_options.keys())
            
            total = len(selected)
            
            # Generate and save each visualization
            for i, viz_key in enumerate(selected):
                # Update progress
                progress = (i / total) * 100
                self.queue.put(("update_progress", progress))
                self.queue.put(("update_status", f"Ukládám: {viz_key}..."))
                
                # Generate visualization
                fig = self.create_visualization(viz_key)
                
                # Save to file
                filepath = os.path.join(output_dir, f"{viz_key}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # Final update
            self.queue.put(("update_progress", 100))
            self.queue.put(("update_status", "Uložení dokončeno"))
            self.queue.put(("enable_buttons", None))
            
            # Show success message
            self.queue.put(("show_info", f"Grafy byly úspěšně uloženy do složky:\n{output_dir}"))
            
        except Exception as e:
            logger.error(f"Chyba při ukládání vizualizací: {e}")
            self.queue.put(("error", str(e)))

# Run the application
if __name__ == "__main__":
    app = NewsVisualizationApp()
    app.mainloop()