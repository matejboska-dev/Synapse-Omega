import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
import logging
from tqdm import tqdm
import re

# logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    class for training and using a model that analyzes sentiment in news articles
    """
    
    def __init__(self, classifier_type='logistic_regression'):
        """
        initialize the sentiment analyzer
        
        args:
            classifier_type (str): type of classifier to use ('logistic_regression', 'svm', or 'random_forest')
        """
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.model = None
        self.pipeline = None
        
        # sentiment labels
        self.labels = ['negative', 'neutral', 'positive']
        
    def _select_classifier(self):
        """
        select the appropriate classifier based on classifier_type
        
        returns:
            sklearn classifier: the selected classifier object
        """
        if self.classifier_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        elif self.classifier_type == 'svm':
            return LinearSVC(C=1.0, max_iter=10000)
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, n_jobs=-1)
        else:
            logger.warning(f"unknown classifier type: {self.classifier_type}, defaulting to logistic_regression")
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')

    def extract_sentiment_features(self, texts):
        """
        extract additional features related to sentiment
        
        args:
            texts (array-like): input text data
            
        returns:
            pandas.DataFrame: extracted features
        """
        features = pd.DataFrame()
        
        # czech sentiment words
        positive_words = [
            'dobrý', 'skvělý', 'výborný', 'pozitivní', 'úspěch', 'radost', 'krásný', 'příjemný',
            'štěstí', 'spokojený', 'výhra', 'zisk', 'růst', 'lepší', 'nejlepší', 'zlepšení',
            'výhoda', 'prospěch', 'podpora', 'rozvoj', 'pokrok', 'úspěšný', 'optimistický',
            'šťastný', 'veselý', 'bezpečný', 'klidný', 'prospěšný', 'úžasný', 'perfektní'
        ]
        
        negative_words = [
            'špatný', 'negativní', 'problém', 'potíž', 'selhání', 'prohra', 'ztráta', 'pokles',
            'krize', 'konflikt', 'smrt', 'válka', 'nehoda', 'tragédie', 'nebezpečí', 'zhoršení',
            'škoda', 'nízký', 'horší', 'nejhorší', 'slabý', 'nepříznivý', 'riziko', 'hrozba',
            'kritický', 'závažný', 'obtížný', 'těžký', 'násilí', 'strach', 'obavy', 'útok'
        ]
        
        # convert texts to list if it's a series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # count positive and negative words
        features['positive_word_count'] = [
            sum(1 for word in re.findall(r'\b\w+\b', text.lower()) if word in positive_words) 
            for text in tqdm(texts, desc="counting positive words")
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in re.findall(r'\b\w+\b', text.lower()) if word in negative_words)
            for text in tqdm(texts, desc="counting negative words")
        ]
        
        # calculate ratio
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        # text length features
        features['text_length'] = [len(text) for text in texts]
        features['word_count'] = [len(text.split()) for text in texts]
        
        return features
    
    def auto_label_data(self, texts, threshold=0.5):
        """
        automatically generate sentiment labels based on simple word counting
        this is a helper method for when no labeled data is available
        
        args:
            texts (array-like): input text data
            threshold (float): threshold for positive/negative classification
            
        returns:
            array: sentiment labels (0 = negative, 1 = neutral, 2 = positive)
        """
        features = self.extract_sentiment_features(texts)
        
        # determine sentiment based on ratio
        sentiment = np.zeros(len(texts), dtype=int)
        
        # neutral if ratio is between 1-threshold and 1+threshold
        sentiment[(features['sentiment_ratio'] >= 1-threshold) & 
                  (features['sentiment_ratio'] <= 1+threshold)] = 1
        
        # positive if ratio is above 1+threshold
        sentiment[features['sentiment_ratio'] > 1+threshold] = 2
        
        # negative if ratio is below 1-threshold
        sentiment[features['sentiment_ratio'] < 1-threshold] = 0
        
        return sentiment
        
    def fit(self, X, y=None, test_size=0.2, random_state=42):
        """
        train the sentiment analyzer on the given data
        if y is None, auto-generate labels using keyword-based approach
        
        args:
            X (array-like): input text data (article content)
            y (array-like, optional): target labels (0 = negative, 1 = neutral, 2 = positive)
            test_size (float): proportion of data to use for testing
            random_state (int): random seed for reproducibility
            
        returns:
            dict: training results including classification report
        """
        # auto-generate labels if none provided
        if y is None:
            logger.info("no labels provided, auto-generating sentiment labels...")
            y = self.auto_label_data(X)
            logger.info(f"auto-generated labels: {np.bincount(y)} (negative, neutral, positive)")
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # create classifier
        classifier = self._select_classifier()
        
        # create and train pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', classifier)
        ])
        
        logger.info(f"training {self.classifier_type} classifier with {len(X_train)} samples...")
        self.pipeline.fit(X_train, y_train)
        
        # evaluate model
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # generate classification report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.labels,
            output_dict=True
        )
        
        # log results
        logger.info(f"model trained with accuracy: {accuracy:.4f}")
        
        # return results
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'class_distribution': np.bincount(y).tolist(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict(self, texts):
        """
        predict sentiment for given texts
        
        args:
            texts (array-like): input text data
            
        returns:
            array: predicted sentiment labels (0 = negative, 1 = neutral, 2 = positive)
        """
        if not self.pipeline:
            raise ValueError("model not trained yet. call fit() first.")
        
        # predict sentiment
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts):
        """
        predict probability distributions over sentiment classes for given texts
        
        args:
            texts (array-like): input text data
            
        returns:
            array: probability distributions (only works with logistic_regression)
        """
        if not self.pipeline:
            raise ValueError("model not trained yet. call fit() first.")
        
        if not hasattr(self.pipeline['classifier'], 'predict_proba'):
            raise ValueError("this classifier doesn't support probability predictions.")
        
        # predict probability distributions
        return self.pipeline.predict_proba(texts)
    
    def save_model(self, model_dir='models/sentiment_analyzer'):
        """
        save the trained model to disk
        
        args:
            model_dir (str): directory to save the model
        """
        if not self.pipeline:
            raise ValueError("model not trained yet. call fit() first.")
        
        # create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # save pipeline
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # save model info
        model_info = {
            'classifier_type': self.classifier_type,
            'labels': self.labels
        }
        
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir='models/sentiment_analyzer'):
        """
        load a trained model from disk
        
        args:
            model_dir (str): directory with saved model files
            
        returns:
            SentimentAnalyzer: loaded model
        """
        # load model info
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # create instance
        instance = cls(classifier_type=model_info['classifier_type'])
        
        # load pipeline
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
            instance.pipeline = pickle.load(f)
        
        # set labels
        if 'labels' in model_info:
            instance.labels = model_info['labels']
        
        logger.info(f"model loaded from {model_dir}")
        
        return instance