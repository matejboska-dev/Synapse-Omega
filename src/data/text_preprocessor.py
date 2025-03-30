import re
import unicodedata
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import logging
import string

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    """
    Enhanced class for preprocessing text data for NLP tasks.
    """
    
    def __init__(self, language='czech', remove_stopwords=True, stemming=False, 
                 lemmatization=False, remove_accents=True, min_word_length=2):
        """
        Initialize the text preprocessor.
        
        Args:
            language (str): Language for stopwords ('czech' or 'english')
            remove_stopwords (bool): Whether to remove stopwords
            stemming (bool): Whether to apply stemming
            lemmatization (bool): Whether to apply lemmatization (if available)
            remove_accents (bool): Whether to remove accents from text
            min_word_length (int): Minimum length of words to keep
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.remove_accents = remove_accents
        self.min_word_length = min_word_length
        
        # Initialize word tokenizer with fallback to simple splitting
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        # Initialize stemmer
        if self.stemming:
            try:
                if language == 'czech':
                    # Czech is not supported, use English stemmer
                    logging.warning(f"Stemmer not available for language: {language}. Using 'english' instead.")
                    self.stemmer = SnowballStemmer('english')
                    self.stemming_language = 'english'
                else:
                    # Try with specified language
                    self.stemmer = SnowballStemmer(language)
                    self.stemming_language = language
            except ValueError as e:
                logging.warning(f"Stemmer initialization failed: {e}. Disabling stemming.")
                self.stemming = False
        
        # Initialize stopwords
        if self.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words(language))
                # Add custom stopwords for Czech news
                if language == 'czech':
                    self.stop_words.update([
                        'podle', 'proto', 'nové', 'jeho', 'které', 'také', 'jsme', 'mezi', 
                        'může', 'řekl', 'uvedl', 'další', 'této', 'byly', 'bude', 'byla', 
                        'jako', 'více', 'však', 'když', 'pokud', 'aby', 'již', 'let', 'tak',
                        'při', 'jen', 'ale', 'dnes', 'ještě', 'není', 'kde', 'což', 'která',
                        'své', 'svůj', 'svou', 'mimo', 'toho', 'tedy', 'tím', 'tam', 'pak',
                        'tento', 'tato', 'tyto', 'toto', 'může', 'například', 'uvádí', 'uvádějí',
                        'nyní', 'protože', 'sdělil', 'informoval', 'měl', 'měla', 'uvedla', 'ale',
                        'lidí', 'letech', 'korun', 'roce', 'roku', 'řekla', 'řekli', 'vůbec',
                        'pouze', 'právě', 'kvůli', 'včera', 'vždy', 'neboť', 'kromě', 'přitom',
                        'zprávy', 'zpráva', 'článek', 'článku', 'řekl', 'řekla', 'dodal', 'dodala',
                        'doplnil', 'doplnila', 'prohlásil', 'prohlásila', 'poznamenal', 'uvedla',
                        'reagoval', 'reagovala', 'napsal', 'napsala', 'podotkl', 'podotkla',
                        'upozornil', 'upozornila', 'tvrdí', 'tvrdil', 'tvrdila', 'oznámil',
                        'oznámila', 'informoval', 'informovala', 'potvrdil', 'potvrdila', 'připomenul',
                        'připomněla', 'zdůraznil', 'zdůraznila'
                    ])
            except OSError:
                logging.warning(f"Stopwords not available for language: {language}. Using empty set.")
                self.stop_words = set()
        else:
            self.stop_words = set()
    
    def remove_accents_from_text(self, text):
        """
        Remove accents from text (convert 'ě' to 'e', 'č' to 'c', etc.)
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without accents
        """
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                      if not unicodedata.combining(c))
    
    def simple_tokenize(self, text):
        """Simple tokenization as fallback"""
        return self.tokenizer.tokenize(text)
    
    def preprocess_text(self, text):
        """
        Preprocess text by applying multiple operations:
        - Convert to lowercase
        - Remove accents (optional)
        - Remove URLs, emails, numbers, special characters
        - Remove punctuation
        - Tokenize
        - Remove stopwords (optional)
        - Apply stemming (optional)
        - Remove short words
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove accents
        if self.remove_accents:
            text = self.remove_accents_from_text(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers with context (e.g., "25 let", "100 Kč")
        text = re.sub(r'\b\d+\s+\w+\b', '', text)
        
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize - use simple tokenization to avoid NLTK errors
        try:
            tokens = self.simple_tokenize(text)
        except Exception as e:
            logging.warning(f"Tokenization error: {e}. Using simple split.")
            tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming - only if enabled and successfully initialized
        if self.stemming:
            try:
                tokens = [self.stemmer.stem(token) for token in tokens]
            except Exception as e:
                logging.warning(f"Stemming error: {e}. Skipping stemming.")
        
        # Remove short words
        tokens = [token for token in tokens if len(token) >= self.min_word_length]
        
        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)
        
        # Remove extra whitespace
        preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text).strip()
        
        return preprocessed_text
    
    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess text in a DataFrame column.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            text_column (str): Name of the column containing text to preprocess
            
        Returns:
            pandas.DataFrame: DataFrame with preprocessed text
        """
        if text_column not in df.columns:
            logging.warning(f"Column '{text_column}' not found in DataFrame.")
            return df
        
        # Create copy of DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Create new column for preprocessed text
        preprocessed_column = f"{text_column}_preprocessed"
        
        # Apply preprocessing to each row
        df_copy[preprocessed_column] = df_copy[text_column].apply(
            lambda x: self.preprocess_text(x) if pd.notnull(x) else ""
        )
        
        return df_copy
    
    def get_document_features(self, text):
        """
        Extract additional features from text document
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of extracted features
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'unique_word_count': 0,
                'lexical_diversity': 0
            }
        
        # Basic counts - use simple tokenization
        try:
            words = self.simple_tokenize(text.lower())
        except:
            words = text.lower().split()
            
        word_count = len(words)
        char_count = len(text)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Unique words
        unique_words = set(words)
        unique_word_count = len(unique_words)
        
        # Lexical diversity (ratio of unique words to total words)
        lexical_diversity = unique_word_count / max(word_count, 1)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'unique_word_count': unique_word_count,
            'lexical_diversity': lexical_diversity
        }