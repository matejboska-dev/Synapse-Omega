import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import pickle
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for extracting text features
    """
    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            use_idf=True,
            analyzer='word'
        )
        self.char_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(2, 6),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True,
            analyzer='char'
        )
        
    def fit(self, X, y=None):
        self.word_vectorizer.fit(X)
        self.char_vectorizer.fit(X)
        return self
        
    def transform(self, X):
        word_features = self.word_vectorizer.transform(X)
        char_features = self.char_vectorizer.transform(X)
        # Combine both feature matrices
        return np.hstack([
            word_features.toarray(),
            char_features.toarray()
        ])

class EnhancedCategoryClassifier:
    """
    Enhanced class for training and using a model that classifies news articles into categories
    with improved preprocessing and feature extraction
    """
    
    def __init__(self, classifier_type='ensemble', min_samples_per_category=10, max_categories=None):
        """
        Initialize the classifier
        
        Args:
            classifier_type (str): Type of classifier to use ('naive_bayes', 'logistic_regression', 
                                   'random_forest', 'svm', 'gradient_boosting', 'ensemble')
            min_samples_per_category (int): Minimum number of samples required for a category
            max_categories (int, optional): Maximum number of top categories to include
        """
        self.classifier_type = classifier_type
        self.min_samples_per_category = min_samples_per_category
        self.max_categories = max_categories
        self.label_encoder = LabelEncoder()
        self.pipeline = None
        self.categories = None
        self.feature_extractor = TextFeatureExtractor()
    def _select_classifier(self):
        """
        Select the appropriate classifier based on classifier_type
        
        Returns:
            sklearn classifier: The selected classifier object
        """
        if self.classifier_type == 'naive_bayes':
            return MultinomialNB()
        elif self.classifier_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_split=5, 
                                        min_samples_leaf=2, n_jobs=-1)
        elif self.classifier_type == 'svm':
            return LinearSVC(C=1.0, max_iter=10000)
        elif self.classifier_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, 
                                            max_depth=7, min_samples_split=5)
        elif self.classifier_type == 'ensemble':
            # Creating a voting classifier is more complex and would need to be handled separately
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        else:
            logger.warning(f"Unknown classifier type: {self.classifier_type}, defaulting to logistic_regression")
            return LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')

    def preprocess_categories(self, categories):
        """
        Preprocess categories by consolidating similar ones and filtering rare ones
        
        Args:
            categories (pandas.Series): Series of category labels
            
        Returns:
            pandas.Series: Preprocessed categories
        """
        # Count occurrences of each category
        category_counts = categories.value_counts()
        
        # Filter categories with too few samples
        valid_categories = category_counts[category_counts >= self.min_samples_per_category].index.tolist()
        
        # Limit to top N categories if specified
        if self.max_categories and len(valid_categories) > self.max_categories:
            valid_categories = category_counts.nlargest(self.max_categories).index.tolist()
        
        # Replace rare categories with 'Other'
        processed_categories = categories.copy()
        processed_categories[~processed_categories.isin(valid_categories)] = 'Other'
        
        # Store the list of valid categories
        self.categories = valid_categories + (['Other'] if len(valid_categories) < len(category_counts) else [])
        
        return processed_categories
    
    def perform_grid_search(self, X_train, y_train, param_grid=None):
        """
        Perform grid search to find optimal hyperparameters
        
        Args:
            X_train: Training data
            y_train: Training labels
            param_grid: Dictionary of parameters for grid search
            
        Returns:
            dict: Best parameters
        """
        if param_grid is None:
            if self.classifier_type == 'naive_bayes':
                param_grid = {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
                    'fit_prior': [True, False]
                }
            elif self.classifier_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 0.5, 1.0, 5.0, 10.0],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000]
                }
            elif self.classifier_type == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [30, 50, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.classifier_type == 'svm':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'max_iter': [5000, 10000]
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
    
    def fit(self, X, y, test_size=0.2, random_state=42, perform_grid_search=False):
        """
        Train the classifier on the given data
        
        Args:
            X (array-like): Input text data (article content)
            y (array-like): Target labels (categories)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            perform_grid_search (bool): Whether to perform grid search for hyperparameter tuning
            
        Returns:
            dict: Training results including classification report
        """
        # Preprocess categories
        processed_y = self.preprocess_categories(pd.Series(y))
        
        # Encode category labels
        encoded_y = self.label_encoder.fit_transform(processed_y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_y, test_size=test_size, random_state=random_state, stratify=encoded_y
        )
        
        # Transform features
        logger.info("Extracting features...")
        self.feature_extractor.fit(X_train)
        X_train_transformed = self.feature_extractor.transform(X_train)
        
        # Create classifier, possibly with grid search
        if perform_grid_search:
            best_params, best_classifier = self.perform_grid_search(X_train_transformed, y_train)
            classifier = best_classifier
        else:
            classifier = self._select_classifier()
        
        # Train classifier
        logger.info(f"Training {self.classifier_type} classifier with {len(X_train)} samples...")
        classifier.fit(X_train_transformed, y_train)
        
        # Save trained classifier
        self.classifier = classifier
        
        # Evaluate model
        X_test_transformed = self.feature_extractor.transform(X_test)
        y_pred = classifier.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Log results
        logger.info(f"Model trained with accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
        
        # Create confusion matrix visualization
        cm = confusion_matrix(y_test, y_pred)
        
        # Return results
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'class_names': self.label_encoder.classes_.tolist(),
            'confusion_matrix': cm.tolist()
        }
    
    def predict(self, texts):
        """
        Predict categories for given texts
        
        Args:
            texts (array-like): Input text data
            
        Returns:
            array: Predicted category labels
        """
        if not hasattr(self, 'classifier'):
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Transform features
        X_transformed = self.feature_extractor.transform(texts)
        
        # Predict category indices
        pred_indices = self.classifier.predict(X_transformed)
        
        # Convert indices to category labels
        return self.label_encoder.inverse_transform(pred_indices)
    
    def predict_proba(self, texts):
        """
        Predict probability distributions over categories for given texts
        
        Args:
            texts (array-like): Input text data
            
        Returns:
            array: Probability distributions (if classifier supports it)
        """
        if not hasattr(self, 'classifier'):
            raise ValueError("Model not trained yet. Call fit() first.")
        
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError("This classifier doesn't support probability predictions.")
        
        # Transform features
        X_transformed = self.feature_extractor.transform(texts)
        
        # Predict probability distributions
        return self.classifier.predict_proba(X_transformed)
    
    def save_model(self, model_dir='models/enhanced_category_classifier'):
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
        
        # Save feature extractor
        with open(os.path.join(model_dir, 'feature_extractor.pkl'), 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        
        # Save label encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save categories
        with open(os.path.join(model_dir, 'categories.pkl'), 'wb') as f:
            pickle.dump(self.categories, f)
        
        # Save model info
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
    def load_model(cls, model_dir='models/enhanced_category_classifier'):
        """
        Load a trained model from disk
        
        Args:
            model_dir (str): Directory with saved model files
            
        Returns:
            EnhancedCategoryClassifier: Loaded model
        """
        # Load model info
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # Create instance
        instance = cls(
            classifier_type=model_info['classifier_type'],
            min_samples_per_category=model_info['min_samples_per_category'],
            max_categories=model_info['max_categories']
        )
        
        # Load classifier
        with open(os.path.join(model_dir, 'classifier.pkl'), 'rb') as f:
            instance.classifier = pickle.load(f)
        
        # Load feature extractor
        with open(os.path.join(model_dir, 'feature_extractor.pkl'), 'rb') as f:
            instance.feature_extractor = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            instance.label_encoder = pickle.load(f)
        
        # Load categories
        with open(os.path.join(model_dir, 'categories.pkl'), 'rb') as f:
            instance.categories = pickle.load(f)
        
        logger.info(f"Model loaded from {model_dir} with {model_info['num_categories']} categories")
        
        return instance