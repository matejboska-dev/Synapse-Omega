import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
import logging
from tqdm import tqdm

# logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CategoryClassifier:
    """
    class for training and using a model that classifies news articles into categories
    """
    
    def __init__(self, classifier_type='naive_bayes', min_samples_per_category=10, max_categories=None):
        """
        initialize the classifier
        
        args:
            classifier_type (str): type of classifier to use ('naive_bayes', 'logistic_regression', or 'random_forest')
            min_samples_per_category (int): minimum number of samples required for a category to be included
            max_categories (int, optional): maximum number of top categories to include
        """
        self.classifier_type = classifier_type
        self.min_samples_per_category = min_samples_per_category
        self.max_categories = max_categories
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.model = None
        self.pipeline = None
        self.categories = None
        
    def _select_classifier(self):
        """
        select the appropriate classifier based on classifier_type
        
        returns:
            sklearn classifier: the selected classifier object
        """
        if self.classifier_type == 'naive_bayes':
            return MultinomialNB()
        elif self.classifier_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, n_jobs=-1)
        else:
            logger.warning(f"Unknown classifier type: {self.classifier_type}, defaulting to naive_bayes")
            return MultinomialNB()

    def preprocess_categories(self, categories):
        """
        preprocess categories by consolidating similar ones and filtering rare ones
        
        args:
            categories (pandas.Series): series of category labels
            
        returns:
            pandas.Series: preprocessed categories
        """
        # count occurrences of each category
        category_counts = categories.value_counts()
        
        # filter categories with too few samples
        valid_categories = category_counts[category_counts >= self.min_samples_per_category].index.tolist()
        
        # limit to top N categories if specified
        if self.max_categories and len(valid_categories) > self.max_categories:
            valid_categories = category_counts.nlargest(self.max_categories).index.tolist()
        
        # replace rare categories with 'Other'
        processed_categories = categories.copy()
        processed_categories[~processed_categories.isin(valid_categories)] = 'Other'
        
        # store the list of valid categories
        self.categories = valid_categories + (['Other'] if len(valid_categories) < len(category_counts) else [])
        
        return processed_categories
    
    def fit(self, X, y, test_size=0.2, random_state=42):
        """
        train the classifier on the given data
        
        args:
            X (array-like): input text data (article content)
            y (array-like): target labels (categories)
            test_size (float): proportion of data to use for testing
            random_state (int): random seed for reproducibility
            
        returns:
            dict: training results including classification report
        """
        # preprocess categories
        processed_y = self.preprocess_categories(pd.Series(y))
        
        # encode category labels
        encoded_y = self.label_encoder.fit_transform(processed_y)
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_y, test_size=test_size, random_state=random_state, stratify=encoded_y
        )
        
        # create classifier
        classifier = self._select_classifier()
        
        # create and train pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', classifier)
        ])
        
        logger.info(f"Training {self.classifier_type} classifier with {len(X_train)} samples...")
        self.pipeline.fit(X_train, y_train)
        
        # evaluate model
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # generate classification report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # log results
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # return results
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'class_names': self.label_encoder.classes_.tolist(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def predict(self, texts):
        """
        predict categories for given texts
        
        args:
            texts (array-like): input text data
            
        returns:
            array: predicted category labels
        """
        if not self.pipeline:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # predict category indices
        pred_indices = self.pipeline.predict(texts)
        
        # convert indices to category labels
        return self.label_encoder.inverse_transform(pred_indices)
    
    def predict_proba(self, texts):
        """
        predict probability distributions over categories for given texts
        
        args:
            texts (array-like): input text data
            
        returns:
            array: probability distributions
        """
        if not self.pipeline:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # predict probability distributions
        return self.pipeline.predict_proba(texts)
    
    def save_model(self, model_dir='models/category_classifier'):
        """
        save the trained model to disk
        
        args:
            model_dir (str): directory to save the model
        """
        if not self.pipeline:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # save pipeline
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        # save label encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # save categories
        with open(os.path.join(model_dir, 'categories.pkl'), 'wb') as f:
            pickle.dump(self.categories, f)
        
        # save model info
        model_info = {
            'classifier_type': self.classifier_type,
            'min_samples_per_category': self.min_samples_per_category,
            'max_categories': self.max_categories,
            'num_categories': len(self.label_encoder.classes_),
            'category_names': self.label_encoder.classes_.tolist()
        }
        
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir='models/category_classifier'):
        """
        load a trained model from disk
        
        args:
            model_dir (str): directory with saved model files
            
        returns:
            CategoryClassifier: loaded model
        """
        # load model info
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # create instance
        instance = cls(
            classifier_type=model_info['classifier_type'],
            min_samples_per_category=model_info['min_samples_per_category'],
            max_categories=model_info['max_categories']
        )
        
        # load pipeline
        with open(os.path.join(model_dir, 'pipeline.pkl'), 'rb') as f:
            instance.pipeline = pickle.load(f)
        
        # load label encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            instance.label_encoder = pickle.load(f)
        
        # load categories
        with open(os.path.join(model_dir, 'categories.pkl'), 'rb') as f:
            instance.categories = pickle.load(f)
        
        logger.info(f"Model loaded from {model_dir} with {model_info['num_categories']} categories")
        
        return instance