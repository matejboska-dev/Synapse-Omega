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
    class for analyzing and visualizing article data
    """
    
    def __init__(self, data):
        """
        initialize with dataframe containing article data
        
        args:
            data (pandas.DataFrame): dataframe with article data
        """
        self.data = data
        self.stats = {}
        
    def compute_basic_stats(self):
        """
        compute basic statistics about the data
        """
        logger.info("computing basic statistics about the data")
        
        # check if dataframe contains expected columns
        expected_columns = ['Title', 'Content', 'Source', 'Category', 'PublishDate']
        for col in expected_columns:
            if col not in self.data.columns:
                logger.warning(f"column '{col}' not in data")
        
        # number of articles
        self.stats['total_articles'] = len(self.data)
        
        # statistics by source (if Source column exists)
        if 'Source' in self.data.columns:
            source_counts = self.data['Source'].value_counts()
            self.stats['articles_by_source'] = source_counts.to_dict()
            self.stats['num_sources'] = len(source_counts)
        
        # statistics by category (if Category column exists)
        if 'Category' in self.data.columns:
            category_counts = self.data['Category'].value_counts()
            self.stats['articles_by_category'] = category_counts.to_dict()
            self.stats['num_categories'] = len(category_counts)
        
        # time statistics (if PublishDate column exists)
        if 'PublishDate' in self.data.columns:
            # convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(self.data['PublishDate']):
                try:
                    self.data['PublishDate'] = pd.to_datetime(self.data['PublishDate'])
                except Exception as e:
                    logger.error(f"error converting date: {str(e)}")
            
            # date range
            self.stats['date_range'] = {
                'min': self.data['PublishDate'].min().strftime('%Y-%m-%d'),
                'max': self.data['PublishDate'].max().strftime('%Y-%m-%d')
            }
            
            # articles by day of week
            self.data['DayOfWeek'] = self.data['PublishDate'].dt.day_name()
            day_counts = self.data['DayOfWeek'].value_counts()
            self.stats['articles_by_day'] = day_counts.to_dict()
        
        # content statistics
        if 'Content' in self.data.columns:
            # content length
            self.data['ContentLength'] = self.data['Content'].apply(lambda x: len(str(x)))
            self.stats['content_length'] = {
                'mean': self.data['ContentLength'].mean(),
                'median': self.data['ContentLength'].median(),
                'min': self.data['ContentLength'].min(),
                'max': self.data['ContentLength'].max()
            }
            
            # word count
            self.data['WordCount'] = self.data['Content'].apply(lambda x: len(str(x).split()))
            self.stats['word_count'] = {
                'mean': self.data['WordCount'].mean(),
                'median': self.data['WordCount'].median(),
                'min': self.data['WordCount'].min(),
                'max': self.data['WordCount'].max()
            }
        # if we already have ArticleLength and WordCount columns, use them
        elif 'ArticleLength' in self.data.columns:
            self.stats['content_length'] = {
                'mean': self.data['ArticleLength'].mean(),
                'median': self.data['ArticleLength'].median(),
                'min': self.data['ArticleLength'].min(),
                'max': self.data['ArticleLength'].max()
            }
            
            if 'WordCount' in self.data.columns:
                self.stats['word_count'] = {
                    'mean': self.data['WordCount'].mean(),
                    'median': self.data['WordCount'].median(),
                    'min': self.data['WordCount'].min(),
                    'max': self.data['WordCount'].max()
                }
        
        logger.info("basic statistics computed")
        return self.stats
    
    def visualize_basic_stats(self, output_dir='reports/figures'):
        """
        create basic visualizations and save them to output directory
        
        args:
            output_dir (str): directory for saving images
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"creating visualizations in directory {output_dir}")
        
        # set style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. articles by source
        if 'Source' in self.data.columns:
            plt.figure(figsize=(12, 6))
            source_counts = self.data['Source'].value_counts().head(15)  # top 15 sources
            ax = sns.barplot(x=source_counts.index, y=source_counts.values)
            plt.title('Number of articles by source (top 15)')
            plt.xlabel('Source')
            plt.ylabel('Number of articles')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_source.png")
            plt.close()
        
        # 2. articles by category
        if 'Category' in self.data.columns:
            plt.figure(figsize=(12, 6))
            category_counts = self.data['Category'].value_counts().head(15)  # top 15 categories
            ax = sns.barplot(x=category_counts.index, y=category_counts.values)
            plt.title('Number of articles by category (top 15)')
            plt.xlabel('Category')
            plt.ylabel('Number of articles')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_category.png")
            plt.close()
        
        # 3. articles over time
        if 'PublishDate' in self.data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data['PublishDate']):
                try:
                    self.data['PublishDate'] = pd.to_datetime(self.data['PublishDate'])
                except Exception as e:
                    logger.error(f"error converting date: {str(e)}")
                    
            plt.figure(figsize=(14, 6))
            date_counts = self.data.groupby(self.data['PublishDate'].dt.date).size()
            plt.plot(date_counts.index, date_counts.values, marker='o')
            plt.title('Number of articles by date')
            plt.xlabel('Date')
            plt.ylabel('Number of articles')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_date.png")
            plt.close()
        
        # 4. article length distribution
        if 'ContentLength' in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data['ContentLength'], bins=30, kde=True)
            plt.title('Article length distribution')
            plt.xlabel('Article length (characters)')
            plt.ylabel('Number of articles')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/content_length_distribution.png")
            plt.close()
        elif 'ArticleLength' in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data['ArticleLength'], bins=30, kde=True)
            plt.title('Article length distribution')
            plt.xlabel('Article length (characters)')
            plt.ylabel('Number of articles')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/content_length_distribution.png")
            plt.close()
        
        # 5. word count distribution
        if 'WordCount' in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data['WordCount'].clip(upper=1000), bins=30, kde=True)  # clip outliers
            plt.title('Word count distribution in articles')
            plt.xlabel('Word count')
            plt.ylabel('Number of articles')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/word_count_distribution.png")
            plt.close()
        
        # 6. correlation between length and word count
        if 'ContentLength' in self.data.columns and 'WordCount' in self.data.columns:
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x='ContentLength', y='WordCount', data=self.data)
            plt.title('Correlation between article length and word count')
            plt.xlabel('Article length (characters)')
            plt.ylabel('Word count')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/length_words_correlation.png")
            plt.close()
        elif 'ArticleLength' in self.data.columns and 'WordCount' in self.data.columns:
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x='ArticleLength', y='WordCount', data=self.data)
            plt.title('Correlation between article length and word count')
            plt.xlabel('Article length (characters)')
            plt.ylabel('Word count')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/length_words_correlation.png")
            plt.close()
        
        # 7. articles by day of week
        if 'PublishDate' in self.data.columns:
            if not 'DayOfWeek' in self.data.columns:
                self.data['DayOfWeek'] = self.data['PublishDate'].dt.day_name()
                
            plt.figure(figsize=(10, 6))
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = self.data['DayOfWeek'].value_counts().reindex(days_order)
            sns.barplot(x=day_counts.index, y=day_counts.values)
            plt.title('Number of articles by day of week')
            plt.xlabel('Day of week')
            plt.ylabel('Number of articles')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/articles_by_day.png")
            plt.close()
        
        logger.info(f"visualizations created and saved to {output_dir}")