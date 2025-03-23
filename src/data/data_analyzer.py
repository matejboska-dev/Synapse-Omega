import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Třída pro analýzu a vizualizaci dat článků.
    """
    
    def __init__(self, data):
        """
        Inicializace s DataFrame obsahujícím data článků.
        
        Args:
            data (pandas.DataFrame): DataFrame s daty článků
        """
        self.data = data
        self.stats = {}
        
    def compute_basic_stats(self):
        """
        Výpočet základních statistik o datech.
        """
        logger.info("Počítám základní statistiky o datech")
        
        # Kontrola, zda DataFrame obsahuje očekávané sloupce
        expected_columns = ['Title', 'Content', 'Source', 'Category', 'PublishDate']
        for col in expected_columns:
            if col not in self.data.columns:
                logger.warning(f"Sloupec '{col}' není v datech")
        
        # Počet článků
        self.stats['total_articles'] = len(self.data)
        
        # Statistiky podle zdroje (pokud existuje sloupec Source)
        if 'Source' in self.data.columns:
            source_counts = self.data['Source'].value_counts()
            self.stats['articles_by_source'] = source_counts.to_dict()
            self.stats['num_sources'] = len(source_counts)
        
        # Statistiky podle kategorie (pokud existuje sloupec Category)
        if 'Category' in self.data.columns:
            category_counts = self.data['Category'].value_counts()
            self.stats['articles_by_category'] = category_counts.to_dict()
            self.stats['num_categories'] = len(category_counts)
        
        # Časové statistiky (pokud existuje sloupec PublishDate)
        if 'PublishDate' in self.data.columns:
            # Převod na datetime, pokud ještě není
            if not pd.api.types.is_datetime64_any_dtype(self.data['PublishDate']):
                try:
                    self.data['PublishDate'] = pd.to_datetime(self.data['PublishDate'])
                except Exception as e:
                    logger.error(f"Chyba při konverzi data: {str(e)}")
            
            # Časové rozpětí
            self.stats['date_range'] = {
                'min': self.data['PublishDate'].min().strftime('%Y-%m-%d'),
                'max': self.data['PublishDate'].max().strftime('%Y-%m-%d')
            }
            
            # Články podle dne v týdnu
            self.data['DayOfWeek'] = self.data['PublishDate'].dt.day_name()
            day_counts = self.data['DayOfWeek'].value_counts()
            self.stats['articles_by_day'] = day_counts.to_dict()
        
        # Obsahové statistiky
        if 'Content' in self.data.columns:
            # Délka obsahu
            self.data['ContentLength'] = self.data['Content'].apply(lambda x: len(str(x)))
            self.stats['content_length'] = {
                'mean': self.data['ContentLength'].mean(),
                'median': self.data['ContentLength'].median(),
                'min': self.data['ContentLength'].min(),
                'max': self.data['ContentLength'].max()
            }
            
            # Počet slov
            self.data['WordCount'] = self.data['Content'].apply(lambda x: len(str(x).split()))
            self.stats['word_count'] = {
                'mean': self.data['WordCount'].mean(),
                'median': self.data['WordCount'].median(),
                'min': self.data['WordCount'].min(),
                'max': self.data['WordCount'].max()
            }
        
        logger.info("Základní statistiky vypočteny")
        return self.stats
    
    def visualize_basic_stats(self, output_dir='reports/figures'):
        """
        Vytvoření základních vizualizací a jejich uložení do výstupního adresáře.
        
        Args:
            output_dir (str): Adresář pro uložení obrázků
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"Vytvářím vizualizace do adresáře {output_dir}")
        
        # Nastavení stylu
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. Články podle zdroje
        if 'Source' in self.data.columns:
            plt.figure(figsize=(12, 6))
            source_counts = self.data['Source'].value_counts()
            ax = sns.barplot(x=source_counts.index, y=source_counts.values)
            plt.title('Počet článků podle zdroje')
            plt.xlabel('Zdroj')
            plt.ylabel('Počet článků')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_source.png")
            plt.close()
        
        # 2. Články podle kategorie
        if 'Category' in self.data.columns:
            plt.figure(figsize=(12, 6))
            category_counts = self.data['Category'].value_counts()
            ax = sns.barplot(x=category_counts.index, y=category_counts.values)
            plt.title('Počet článků podle kategorie')
            plt.xlabel('Kategorie')
            plt.ylabel('Počet článků')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_category.png")
            plt.close()
        
        # 3. Články v čase
        if 'PublishDate' in self.data.columns:
            plt.figure(figsize=(14, 6))
            date_counts = self.data.groupby(self.data['PublishDate'].dt.date).size()
            plt.plot(date_counts.index, date_counts.values, marker='o')
            plt.title('Počet článků podle data')
            plt.xlabel('Datum')
            plt.ylabel('Počet článků')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_date.png")
            plt.close()
        
        # 4. Distribuce délky článků
        if 'ContentLength' in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data['ContentLength'], bins=30, kde=True)
            plt.title('Distribuce délky článků')
            plt.xlabel('Délka článku (znaky)')
            plt.ylabel('Počet článků')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/content_length_distribution.png")
            plt.close()
        
        # 5. Distribuce počtu slov
        if 'WordCount' in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data['WordCount'], bins=30, kde=True)
            plt.title('Distribuce počtu slov v článcích')
            plt.xlabel('Počet slov')
            plt.ylabel('Počet článků')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/word_count_distribution.png")
            plt.close()
        
        # 6. Korelace délky a počtu slov
        if 'ContentLength' in self.data.columns and 'WordCount' in self.data.columns:
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x='ContentLength', y='WordCount', data=self.data)
            plt.title('Korelace délky článku a počtu slov')
            plt.xlabel('Délka článku (znaky)')
            plt.ylabel('Počet slov')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/length_words_correlation.png")
            plt.close()
        
        # 7. Články podle dne v týdnu
        if 'DayOfWeek' in self.data.columns:
            plt.figure(figsize=(10, 6))
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = self.data['DayOfWeek'].value_counts().reindex(days_order)
            sns.barplot(x=day_counts.index, y=day_counts.values)
            plt.title('Počet článků podle dne v týdnu')
            plt.xlabel('Den v týdnu')
            plt.ylabel('Počet článků')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_day.png")
            plt.close()
        
        logger.info(f"Vizualizace vytvořeny a uloženy do {output_dir}")