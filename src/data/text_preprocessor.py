import pandas as pd
import re
import unicodedata
import logging
from tqdm import tqdm
import nltk
import string

# logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    class for preprocessing article texts
    """
    
    def __init__(self, language='czech'):
        """
        initialize preprocessor
        
        args:
            language (str): text language ('czech' or 'english')
        """
        self.language = language
        
        # download required nltk data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("downloading nltk punkt tokenizer")
            nltk.download('punkt')
            
        try:
            nltk.data.find(f'corpora/stopwords')
        except LookupError:
            logger.info("downloading nltk stopwords")
            nltk.download('stopwords')
        
        # load stopwords for the specified language
        self.stop_words = set()
        try:
            if language == 'czech':
                # for czech language we need to use stopwords from external source
                # because nltk doesn't have czech stopwords directly
                self.load_czech_stopwords()
            else:
                self.stop_words = set(nltk.corpus.stopwords.words(language))
        except Exception as e:
            logger.warning(f"error loading stopwords: {str(e)}")
    
    def load_czech_stopwords(self):
        """
        load czech stopwords from custom source
        """
        # basic czech stopwords
        czech_stopwords = [
            "a", "aby", "ale", "ani", "ano", "asi", "až", "bez", "bude", "budem",
            "budeš", "by", "byl", "byla", "byli", "bylo", "být", "co", "což", "či",
            "další", "do", "ho", "i", "já", "jak", "jako", "je", "jeho", "jej",
            "její", "jejich", "jen", "ještě", "ji", "jich", "jimi", "jinou", "jiný",
            "již", "jsem", "jsi", "jsme", "jsou", "jste", "k", "kam", "kde", "kdo",
            "když", "ke", "která", "které", "kteří", "který", "ku", "má", "mají",
            "máme", "máš", "mé", "mezi", "mi", "mít", "mně", "mnou", "můj", "my",
            "na", "nad", "nám", "námi", "naše", "naši", "ne", "nebo", "neboť",
            "něj", "nějaký", "nelze", "není", "než", "ni", "nic", "nich", "ním",
            "no", "nás", "ný", "o", "od", "on", "ona", "oni", "ono", "onu", "pak",
            "po", "pod", "podle", "pokud", "pouze", "právě", "pro", "proč", "proto",
            "protože", "před", "při", "s", "se", "si", "sice", "své", "svůj", "svých",
            "svým", "svými", "ta", "tak", "také", "takže", "tato", "te", "tě", "tedy",
            "těm", "ten", "tento", "této", "tím", "tímto", "to", "tobě", "tohle",
            "toto", "ty", "týž", "u", "už", "v", "vám", "vámi", "váš", "vaše", "ve",
            "více", "však", "všechen", "vy", "z", "za", "zde", "ze", "že"
        ]
        self.stop_words = set(czech_stopwords)
        logger.info(f"loaded {len(self.stop_words)} czech stopwords")
    
    def clean_text(self, text):
        """
        basic text cleaning
        
        args:
            text (str): input text
        
        returns:
            str: cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # lowercase
        text = text.lower()
        
        # remove html tags
        text = re.sub(r'<.*?>', '', text)
        
        # remove urls
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # remove numbers
        text = re.sub(r'\d+', '', text)
        
        # remove special characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_accents(self, text):
        """
        normalize accents
        
        args:
            text (str): input text
        
        returns:
            str: text with normalized accents
        """
        return unicodedata.normalize('NFKC', text)
    
    def simple_tokenize(self, text):
        """
        simple tokenization by splitting on whitespace
        
        args:
            text (str): input text
        
        returns:
            list: list of tokens (words)
        """
        # simple split by whitespace for languages with whitespace word boundaries
        return text.split()
    
    def remove_stopwords(self, tokens):
        """
        remove stopwords from tokens
        
        args:
            tokens (list): list of tokens
        
        returns:
            list: list of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def preprocess(self, text):
        """
        complete text preprocessing
        
        args:
            text (str): input text
        
        returns:
            list: list of preprocessed tokens
        """
        # clean text
        cleaned_text = self.clean_text(text)
        
        # normalize accents
        normalized_text = self.normalize_accents(cleaned_text)
        
        # tokenize
        tokens = self.simple_tokenize(normalized_text)
        
        # remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        return filtered_tokens
    
    def preprocess_dataframe(self, df, text_column, create_new_column=True):
        """
        preprocess text in dataframe
        
        args:
            df (pandas.DataFrame): input dataframe
            text_column (str): name of text column
            create_new_column (bool): whether to create new column or overwrite existing
        
        returns:
            pandas.DataFrame: dataframe with preprocessed texts
        """
        if text_column not in df.columns:
            logger.error(f"column '{text_column}' not in dataframe")
            return df
        
        # copy dataframe if creating new column
        if create_new_column:
            df_copy = df.copy()
        else:
            df_copy = df
        
        # target column name
        target_column = f"{text_column}_preprocessed" if create_new_column else text_column
        
        # preprocess texts with progress bar
        logger.info(f"preprocessing texts in column '{text_column}'")
        
        # helper function to apply to each row
        def process_row(text):
            tokens = self.preprocess(text)
            return ' '.join(tokens)  # return as joined string for easier further use
        
        # apply to each row with progress bar
        tqdm.pandas(desc="Preprocessing")
        df_copy[target_column] = df_copy[text_column].progress_apply(process_row)
        
        logger.info(f"text preprocessing completed")
        
        return df_copy