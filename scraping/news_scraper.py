import tkinter as tk
from tkinter import ttk, messagebox
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime

class AdvancedNewsAnalysisDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Komplexní Analýza Zpravodajských Článků")
        self.root.geometry("1400x900")

        # Database connection parameters
        self.DB_SERVER = "193.85.203.188"
        self.DB_NAME = "boska"
        self.DB_USER = "boska"
        self.DB_PASSWORD = "123456"

        # Create main layout
        self.create_layout()

        # Load initial data
        self.load_data()

    def connect_to_db(self):
        """Establish a connection to the database."""
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.DB_SERVER};DATABASE={self.DB_NAME};UID={self.DB_USER};PWD={self.DB_PASSWORD}"
        try:
            conn = pyodbc.connect(conn_str)
            return conn
        except Exception as e:
            messagebox.showerror("Chyba databáze", f"Nepodařilo se připojit: {e}")
            return None

    def load_data(self):
        """Load comprehensive data from database."""
        conn = self.connect_to_db()
        if not conn:
            return

        try:
            # Comprehensive query to get detailed insights
            self.df = pd.read_sql("""
                SELECT 
                    SourceName, 
                    Category,
                    PublicationDate,
                    ArticleLength,
                    WordCount,
                    ArticleUrl
                FROM Articles
            """, conn)
            
            conn.close()

            # Data preprocessing
            self.df['PublicationDate'] = pd.to_datetime(self.df['PublicationDate'])
            
            # Populate dropdowns
            self.source_combobox['values'] = sorted(self.df['SourceName'].unique())
            self.category_combobox['values'] = sorted(self.df['Category'].unique())

            # Initial visualization
            self.create_main_dashboard()

        except Exception as e:
            messagebox.showerror("Chyba načítání dat", f"Nepodařilo se načíst data: {e}")

    def create_layout(self):
        """Create the main layout of the dashboard."""
        # Control Frame
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Source Selection
        tk.Label(control_frame, text="Zdroj média:").pack(side=tk.LEFT, padx=5)
        self.source_combobox = ttk.Combobox(control_frame, width=20)
        self.source_combobox.pack(side=tk.LEFT, padx=5)
        self.source_combobox.bind('<<ComboboxSelected>>', self.filter_data)

        # Category Selection
        tk.Label(control_frame, text="Kategorie:").pack(side=tk.LEFT, padx=5)
        self.category_combobox = ttk.Combobox(control_frame, width=20)
        self.category_combobox.pack(side=tk.LEFT, padx=5)
        self.category_combobox.bind('<<ComboboxSelected>>', self.filter_data)

        # Plotting Frame
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_main_dashboard(self):
        """Create a comprehensive dashboard with multiple visualizations."""
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Create a grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Komplexní analýza zpravodajských článků', fontsize=16)

        # 1. Počet článků podle zdrojů
        source_counts = self.df['SourceName'].value_counts()
        axs[0, 0].bar(source_counts.index, source_counts.values, color='skyblue')
        axs[0, 0].set_title('Počet článků podle zdrojů')
        axs[0, 0].set_xticklabels(source_counts.index, rotation=45, ha='right')
        axs[0, 0].set_ylabel('Počet článků')

        # 2. Průměrná délka článků podle zdrojů
        avg_lengths = self.df.groupby('SourceName')['ArticleLength'].mean()
        axs[0, 1].bar(avg_lengths.index, avg_lengths.values, color='lightgreen')
        axs[0, 1].set_title('Průměrná délka článků')
        axs[0, 1].set_xticklabels(avg_lengths.index, rotation=45, ha='right')
        axs[0, 1].set_ylabel('Průměrný počet znaků')

        # 3. Distribuce kategorií
        category_counts = self.df['Category'].value_counts().head(10)
        axs[1, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axs[1, 0].set_title('Top 10 kategorií')

        # 4. Publikační aktivita v čase
        self.df['PublicationMonth'] = self.df['PublicationDate'].dt.to_period('M')
        monthly_counts = self.df.groupby('PublicationMonth').size()
        monthly_counts.plot(kind='line', ax=axs[1, 1], marker='o')
        axs[1, 1].set_title('Publikační aktivita v čase')
        axs[1, 1].set_xlabel('Měsíc')
        axs[1, 1].set_ylabel('Počet článků')

        plt.tight_layout()

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add details table
        self.create_details_table()

    def create_details_table(self):
        """Create a details table with key statistics."""
        details_frame = tk.Frame(self.plot_frame)
        details_frame.pack(fill=tk.X, padx=10, pady=10)

        # Calculate summary statistics
        total_articles = len(self.df)
        unique_sources = self.df['SourceName'].nunique()
        unique_categories = self.df['Category'].nunique()
        avg_article_length = self.df['ArticleLength'].mean()
        date_range = f"{self.df['PublicationDate'].min().date()} - {self.df['PublicationDate'].max().date()}"

        # Create labels with statistics
        stats_labels = [
            f"Celkový počet článků: {total_articles}",
            f"Počet jedinečných zdrojů: {unique_sources}",
            f"Počet kategorií: {unique_categories}",
            f"Průměrná délka článku: {avg_article_length:.2f} znaků",
            f"Časové rozmezí: {date_range}"
        ]

        for i, stat in enumerate(stats_labels):
            tk.Label(details_frame, text=stat, font=('Arial', 10)).grid(row=i//3, column=i%3, padx=10, pady=5, sticky='w')

    def filter_data(self, event=None):
        """Filter data based on selected source and category."""
        selected_source = self.source_combobox.get()
        selected_category = self.category_combobox.get()

        # Apply filters
        filtered_df = self.df.copy()
        if selected_source:
            filtered_df = filtered_df[filtered_df['SourceName'] == selected_source]
        if selected_category:
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]

        # Update visualization with filtered data
        self.show_filtered_details(filtered_df)

    def show_filtered_details(self, filtered_df):
        """Show detailed information about filtered data."""
        # Clear previous details
        for widget in self.plot_frame.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()

        # Create a new top-level window
        details_window = tk.Toplevel(self.root)
        details_window.title("Detaily filtrovaných článků")
        details_window.geometry("800x600")

        # Create Treeview
        columns = ("Zdroj", "Kategorie", "Datum", "Délka", "Slova", "URL")
        tree = ttk.Treeview(details_window, columns=columns, show='headings')

        # Setup column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Insert data
        for _, row in filtered_df.iterrows():
            tree.insert("", "end", values=(
                row['SourceName'], 
                row['Category'], 
                row['PublicationDate'].strftime('%Y-%m-%d %H:%M'), 
                row['ArticleLength'], 
                row['WordCount'], 
                row['ArticleUrl'][:50] + "..." if len(row['ArticleUrl']) > 50 else row['ArticleUrl']
            ))

        # Add scrollbar
        scrollbar = ttk.Scrollbar(details_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Summary statistics for filtered data
        summary_frame = tk.Frame(details_window)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)

        summary_stats = [
            f"Počet článků: {len(filtered_df)}",
            f"Průměrná délka: {filtered_df['ArticleLength'].mean():.2f} znaků",
            f"Průměrný počet slov: {filtered_df['WordCount'].mean():.2f}"
        ]

        for i, stat in enumerate(summary_stats):
            tk.Label(summary_frame, text=stat, font=('Arial', 10)).pack(side=tk.LEFT, padx=10)

def main():
    root = tk.Tk()
    app = AdvancedNewsAnalysisDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()