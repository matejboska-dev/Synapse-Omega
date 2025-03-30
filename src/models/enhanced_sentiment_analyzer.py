import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import pickle
import os
import logging
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    """
    Enhanced class for training and using a model that analyzes sentiment in news articles
    with improved preprocessing, feature extraction, and model selection
    """
    
    def __init__(self, classifier_type='logistic_regression'):
        """
        Initialize the sentiment analyzer
        
        Args:
            classifier_type (str): Type of classifier to use ('logistic_regression', 'svm', 
                                   'random_forest', 'gradient_boosting', 'ensemble')
        """
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(
            max_features=20000, 
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )
        self.char_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(2, 5),
            min_df=2,
            max_df=0.95,
            analyzer='char',
            sublinear_tf=True
        )
        self.model = None
        self.pipeline = None
        
        # Sentiment labels
        self.labels = ['negative', 'neutral', 'positive']
        
        # Enhanced sentiment lexicons
        self._load_sentiment_lexicons()
        
    def _load_sentiment_lexicons(self):
        """
        Load sentiment lexicons for Czech language
        """
        # Define Czech sentiment words (expanded list)
        self.positive_words = [
            'dobrý', 'skvělý', 'výborný', 'pozitivní', 'úspěch', 'radost', 'krásný', 'příjemný',
            'štěstí', 'spokojený', 'výhra', 'zisk', 'růst', 'lepší', 'nejlepší', 'zlepšení',
            'výhoda', 'prospěch', 'podpora', 'rozvoj', 'pokrok', 'úspěšný', 'optimistický',
            'šťastný', 'veselý', 'bezpečný', 'klidný', 'prospěšný', 'úžasný', 'perfektní',
            'vynikající', 'senzační', 'fantastický', 'neuvěřitelný', 'báječný', 'nádherný',
            'velkolepý', 'luxusní', 'přátelský', 'laskavý', 'milý', 'ochotný', 'talentovaný',
            'nadaný', 'inovativní', 'kreativní', 'silný', 'výkonný', 'efektivní', 'užitečný',
            'cenný', 'důležitý', 'ohromující', 'fascinující', 'zajímavý', 'pozoruhodný',
            'inspirativní', 'motivující', 'povzbuzující', 'osvěžující', 'uvolňující',
            'uklidňující', 'příznivý', 'konstruktivní', 'produktivní', 'perspektivní',
            'slibný', 'nadějný', 'obohacující', 'vzrušující', 'úchvatný', 'impozantní', 
            'působivý', 'přesvědčivý', 'vítaný', 'populární', 'oblíbený', 'milovaný',
            'oceňovaný', 'oslavovaný', 'vyzdvihovaný', 'vyžadovaný', 'potřebný', 'žádoucí'
        ]
        
        self.negative_words = [
            'špatný', 'negativní', 'problém', 'potíž', 'selhání', 'prohra', 'ztráta', 'pokles',
            'krize', 'konflikt', 'smrt', 'válka', 'nehoda', 'tragédie', 'nebezpečí', 'zhoršení',
            'škoda', 'nízký', 'horší', 'nejhorší', 'slabý', 'nepříznivý', 'riziko', 'hrozba',
            'kritický', 'závažný', 'obtížný', 'těžký', 'násilí', 'strach', 'obavy', 'útok',
            'katastrofa', 'pohroma', 'neštěstí', 'destrukce', 'zničení', 'zkáza', 'porážka',
            'kolaps', 'pád', 'děsivý', 'hrozný', 'strašný', 'příšerný', 'otřesný', 'hrozivý',
            'znepokojivý', 'alarmující', 'ohavný', 'odpudivý', 'nechutný', 'odporný', 'krutý',
            'brutální', 'agresivní', 'surový', 'barbarský', 'divoký', 'vražedný', 'smrtící',
            'jedovatý', 'toxický', 'škodlivý', 'ničivý', 'zničující', 'fatální', 'smrtelný',
            'zoufalý', 'beznadějný', 'bezmocný', 'deprimující', 'skličující', 'depresivní',
            'smutný', 'bolestný', 'trýznivý', 'traumatický', 'poškozený', 'rozbitý', 'zlomený',
            'naštvaný', 'rozzlobený', 'rozzuřený', 'rozhořčený', 'nenávistný', 'nepřátelský',
            'odmítavý', 'podvodný', 'klamavý', 'lživý', 'falešný', 'neetický', 'nemorální',
            'zkorumpovaný', 'zkažený', 'prohnilý', 'bezcenný', 'zbytečný', 'marný', 'bídný',
            'ubohý', 'žalostný', 'nedostatečný', 'průměrný', 'nudný', 'nezajímavý', 'nezáživný'
        ]
        
    def _select_classifier(self, params=None):
        """
        Select the appropriate classifier based on classifier_type
        
        Args:
            params: Optional parameters for the classifier
        
        Returns:
            sklearn classifier: The selected classifier object
        """
        if params is None:
            params = {}
            
        if self.classifier_type == 'logistic_regression':
            default_params = {'max_iter': 1000, 'C': 1.0, 'solver': 'liblinear', 'class_weight': 'balanced'}
            return LogisticRegression(**{**default_params, **params})
        
        elif self.classifier_type == 'svm':
            default_params = {'C': 1.0, 'max_iter': 10000, 'class_weight': 'balanced'}
            return LinearSVC(**{**default_params, **params})
        
        elif self.classifier_type == 'random_forest':
            default_params = {'n_estimators': 200, 'max_depth': 50, 'min_samples_split': 5, 
                              'min_samples_leaf': 2, 'class_weight': 'balanced', 'n_jobs': -1}
            return RandomForestClassifier(**{**default_params, **params})
        
        elif self.classifier_type == 'gradient_boosting':
            default_params = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 
                              'min_samples_split': 5}
            return GradientBoostingClassifier(**{**default_params, **params})
        
        else:
            logger.warning(f"Unknown classifier type: {self.classifier_type}, defaulting to logistic_regression")
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', class_weight='balanced')

    def extract_sentiment_features(self, texts):
        """
        Extract additional features related to sentiment
        
        Args:
            texts (array-like): Input text data
            
        Returns:
            pandas.DataFrame: Extracted features
        """
        features = pd.DataFrame()
        
        # Convert texts to list if it's a series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Count positive and negative words
        features['positive_word_count'] = [
            sum(1 for word in re.findall(r'\b\w+\b', text.lower()) if word in self.positive_words) 
            for text in tqdm(texts, desc="Counting positive words")
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in re.findall(r'\b\w+\b', text.lower()) if word in self.negative_words)
            for text in tqdm(texts, desc="Counting negative words")
        ]
        
        # Sentiment ratio (positive vs negative)
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        # Text length features
        features['text_length'] = [len(text) for text in texts]
        features['word_count'] = [len(text.split()) for text in texts]
        
        # Average word length
        features['avg_word_length'] = [
            sum(len(word) for word in text.split()) / max(len(text.split()), 1) 
            for text in texts
        ]
        
        # Punctuation counts
        features['exclamation_count'] = [text.count('!') for text in texts]
        features['question_count'] = [text.count('?') for text in texts]
        
        # Uppercase word ratio (could indicate shouting/emphasis)
        features['uppercase_ratio'] = [
            sum(1 for word in text.split() if word.isupper() and len(word) > 1) / max(len(text.split()), 1)
            for text in texts
        ]
        
        return features
    
    def auto_label_data(self, texts, threshold=0.5):
        """
        Automatically generate sentiment labels based on enhanced lexicon-based approach
        
        Args:
            texts (array-like): Input text data
            threshold (float): Threshold for positive/negative classification
            
        Returns:
            array: Sentiment labels (0 = negative, 1 = neutral, 2 = positive)
        """
        features = self.extract_sentiment_features(texts)
        
        # Weighed sentiment score
        pos_weight = 1.0
        neg_weight = 1.2  # Give slightly more weight to negative words
        
        sentiment_scores = pos_weight * features['positive_word_count'] - neg_weight * features['negative_word_count']
        
        # Normalize by text length
        normalized_scores = sentiment_scores / features['word_count'].apply(lambda x: max(x, 10))
        
        # Determine sentiment based on normalized score
        sentiment = np.zeros(len(texts), dtype=int)
        
        # Neutral if score is close to 0 (within threshold)
        sentiment[(normalized_scores >= -threshold) & (normalized_scores <= threshold)] = 1
        
        # Positive if score is above threshold
        sentiment[normalized_scores > threshold] = 2
        
        # Negative if score is below negative threshold
        sentiment[normalized_scores < -threshold] = 0
        
        return sentiment
    
    def perform_grid_search(self, X_train, y_train, param_grid=None):
        """
        Perform grid search to find optimal hyperparameters
        
        Args:
            X_train: Training data features
            y_train: Training labels
            param_grid: Parameter grid for grid search
            
        Returns:
            dict: Best parameters
        """
        if param_grid is None:
            if self.classifier_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000]
                }
            elif self.classifier_type == 'svm':
                param_grid = {
                    'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                    'max_iter': [5000, 10000]
                }
            elif self.classifier_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [30, 50, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.classifier_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                param_grid = {}
        
        if not param_grid:
            logger.info("No parameter grid specified for grid search. Using default parameters.")
            return {}
        
        # Create base classifier
        base_classifier = self._select_classifier()
        
        # Create grid search
        grid_search = GridSearchCV(
            base_classifier,
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        logger.info(f"Performing grid search with {len(X_train)} samples...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_estimator_
        
    def fit(self, X, y=None, test_size=0.2, random_state=42, perform_grid_search=True):
        """
        Train the sentiment analyzer on the given data
        If y is None, auto-generate labels using lexicon-based approach
        
        Args:
            X (array-like): Input text data (article content)
            y (array-like, optional): Target labels (0 = negative, 1 = neutral, 2 = positive)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            perform_grid_search (bool): Whether to perform grid search for hyperparameter tuning
            
        Returns:
            dict: Training results including classification report
        """
        # Auto-generate labels if none provided
        if y is None:
            logger.info("No labels provided, auto-generating sentiment labels...")
            y = self.auto_label_data(X)
            logger.info(f"Auto-generated labels: {np.bincount(y)} (negative, neutral, positive)")
        
        # Extract additional features
        logger.info("Extracting additional sentiment features...")
        additional_features = self.extract_sentiment_features(X)
        
        # Generate TF-IDF features for text
        logger.info("Generating TF-IDF features...")
        word_features = self.vectorizer.fit_transform(X)
        char_features = self.char_vectorizer.fit_transform(X)
        
        # Combine all features
        X_combined = np.hstack([
            word_features.toarray(), 
            char_features.toarray(),
            additional_features.values
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create classifier, possibly with grid search
        if perform_grid_search:
            best_params, best_classifier = self.perform_grid_search(X_train, y_train)
            classifier = best_classifier
        else:
            classifier = self._select_classifier()
        
        # Train model
        logger.info(f"Training {self.classifier_type} classifier with {len(X_train)} samples...")
        classifier.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.labels,
            output_dict=True
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store trained components
        self.classifier = classifier
        
        # Log results
        logger.info(f"Model trained with accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
        
        # Return results
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'class_distribution': np.bincount(y).tolist(),
            'confusion_matrix': cm.tolist(),
            'feature_importances': self._get_feature_importances(classifier, word_features.shape[1], char_features.shape[1], additional_features.shape[1])
        }
    
    def _get_feature_importances(self, classifier, n_word_features, n_char_features, n_additional_features):
        """Get feature importances if the classifier supports it"""
        if hasattr(classifier, 'coef_'):
            # Linear models like LogisticRegression or LinearSVC
            if classifier.coef_.ndim == 2:
                # Multi-class
                importances = np.abs(classifier.coef_).mean(axis=0)
            else:
                # Binary classification
                importances = np.abs(classifier.coef_)
        elif hasattr(classifier, 'feature_importances_'):
            # Tree-based models like RandomForest
            importances = classifier.feature_importances_
        else:
            return {}
        
        # Return information about each type of feature
        return {
            'word_features_importance': float(importances[:n_word_features].mean()),
            'char_features_importance': float(importances[n_word_features:n_word_features+n_char_features].mean()),
            'additional_features_importance': float(importances[-n_additional_features:].mean())
        }
    
    def predict(self, texts):
        """
        Predict sentiment for given texts
        
        Args:
            texts (array-like): Input text data
            
        Returns:
            array: Predicted sentiment labels (0 = negative, 1 = neutral, 2 = positive)
        """
        if not hasattr(self, 'classifier'):
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Extract features
        additional_features = self.extract_sentiment_features(texts)
        word_features = self.vectorizer.transform(texts)
        char_features = self.char_vectorizer.transform(texts)
        
        # Combine features
        X_combined = np.hstack([
            word_features.toarray(), 
            char_features.toarray(),
            additional_features.values
        ])
        
        # Predict sentiment
        return self.classifier.predict(X_combined)
    
    def predict_proba(self, texts):
        """
        Predict probability distributions over sentiment classes for given texts
        
        Args:
            texts (array-like): Input text data
            
        Returns:
            array: Probability distributions (only works with certain classifiers)
        """
        if not hasattr(self, 'classifier'):
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError("This classifier doesn't support probability predictions.")
        
        # Extract features
        additional_features = self.extract_sentiment_features(texts)
        word_features = self.vectorizer.transform(texts)
        char_features = self.char_vectorizer.transform(texts)
        
        # Combine features
        X_combined = np.hstack([
            word_features.toarray(), 
            char_features.toarray(),
            additional_features.values
        ])
        
        # Predict probability distributions
        return self.classifier.predict_proba(X_combined)
    
    def explain_prediction(self, text):
        """
        Explain why a specific text was classified with a particular sentiment
        
        Args:
            text (str): Text to explain
            
        Returns:
            dict: Explanation of sentiment classification
        """
        if not hasattr(self, 'classifier'):
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get prediction
        sentiment_id = self.predict([text])[0]
        sentiment = self.labels[sentiment_id]
        
        # Extract sentiment features
        features = self.extract_sentiment_features([text])
        
        # Find positive and negative words in the text
        positive_words_found = [word for word in re.findall(r'\b\w+\b', text.lower()) 
                               if word in self.positive_words]
        negative_words_found = [word for word in re.findall(r'\b\w+\b', text.lower()) 
                               if word in self.negative_words]
        
        # Calculate sentiment score
        pos_weight = 1.0
        neg_weight = 1.2
        sentiment_score = (pos_weight * features['positive_word_count'].iloc[0] - 
                          neg_weight * features['negative_word_count'].iloc[0])
        
        # Normalize by text length
        word_count = features['word_count'].iloc[0]
        normalized_score = sentiment_score / max(word_count, 10)
        
        # Create explanation
        explanation = {
            'text': text,
            'predicted_sentiment': sentiment,
            'sentiment_score': float(normalized_score),
            'positive_words': positive_words_found[:10],  # Limit to first 10
            'negative_words': negative_words_found[:10],  # Limit to first 10
            'positive_word_count': int(features['positive_word_count'].iloc[0]),
            'negative_word_count': int(features['negative_word_count'].iloc[0]),
            'word_count': int(features['word_count'].iloc[0]),
            'sentiment_ratio': float(features['sentiment_ratio'].iloc[0])
        }
        
        # Add reason based on sentiment
        if sentiment == 'positive':
            if len(positive_words_found) > 0:
                explanation['reason'] = f"Text obsahuje pozitivní slova jako: {', '.join(positive_words_found[:5])}"
            else:
                explanation['reason'] = "Text má celkově pozitivní tón."
        elif sentiment == 'negative':
            if len(negative_words_found) > 0:
                explanation['reason'] = f"Text obsahuje negativní slova jako: {', '.join(negative_words_found[:5])}"
            else:
                explanation['reason'] = "Text má celkově negativní tón."
        else:
            explanation['reason'] = "Text obsahuje vyváženou směs pozitivních a negativních slov nebo neobsahuje dostatek slov s emočním nábojem."
        
        return explanation
    
    def save_model(self, model_dir='models/enhanced_sentiment_analyzer'):
        """
        Save the trained model to disk
        
        Args:
            model_dir (str): Directory to save the model
        """
        if not hasattr(self, 'classifier'):
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save classifier
        with open(os.path.join(model_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Save vectorizers
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(os.path.join(model_dir, 'char_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.char_vectorizer, f)
        
        # Save lexicons
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'wb') as f:
            pickle.dump({
                'positive_words': self.positive_words,
                'negative_words': self.negative_words
            }, f)
        
        # Save model info
        model_info = {
            'classifier_type': self.classifier_type,
            'labels': self.labels
        }
        
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir='models/enhanced_sentiment_analyzer'):
        """
        Load a trained model from disk
        
        Args:
            model_dir (str): Directory with saved model files
            
        Returns:
            EnhancedSentimentAnalyzer: Loaded model
        """
        # Load model info
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # Create instance
        instance = cls(classifier_type=model_info['classifier_type'])
        
        # Load classifier
        with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
            instance.classifier = pickle.load(f)
        
        # Load vectorizers
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            instance.vectorizer = pickle.load(f)
        
        with open(os.path.join(model_dir, 'char_vectorizer.pkl'), 'rb') as f:
            instance.char_vectorizer = pickle.load(f)
        
        # Load lexicons
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
            lexicons = pickle.load(f)
            instance.positive_words = lexicons['positive_words']
            instance.negative_words = lexicons['negative_words']
        
        # Set labels
        if 'labels' in model_info:
            instance.labels = model_info['labels']
        
        logger.info(f"Model loaded from {model_dir}")
        
        return instance