import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_enhanced_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Absolute imports from src directory
from models.enhanced_category_classifier import EnhancedCategoryClassifier
from models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from data.text_preprocessor import TextPreprocessor

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def plot_confusion_matrix(cm, class_names, figsize=(10, 8), cmap='Blues', title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        cm (array): Confusion matrix
        class_names (list): List of class names
        figsize (tuple): Figure size
        cmap (str): Colormap
        title (str): Plot title
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figures_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()

def plot_class_distribution(class_dist, class_names, figsize=(10, 8), title='Class Distribution'):
    """
    Plot class distribution
    
    Args:
        class_dist (array): Class distribution
        class_names (list): List of class names
        figsize (tuple): Figure size
        title (str): Plot title
    """
    plt.figure(figsize=figsize)
    plt.bar(class_names, class_dist)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figures_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()

def plot_feature_importances(importances, title='Feature Importances'):
    """
    Plot feature importances
    
    Args:
        importances (dict): Feature importances
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    features = list(importances.keys())
    values = list(importances.values())
    
    plt.bar(features, values)
    plt.xlabel('Feature Type')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figures_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()

def main():
    """
    Main function for training and evaluating enhanced models
    """
    # Create output directories if they don't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for directory in ['models/enhanced_category_classifier', 'models/enhanced_sentiment_analyzer', 'reports/models']:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensuring directory exists: {dir_path}")
    
    # Load preprocessed data
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'articles_processed.csv')
    
    if not os.path.exists(processed_data_path):
        logger.error(f"Preprocessed data file not found: {processed_data_path}")
        logger.error("Run data_preparation.py first")
        return
    
    logger.info(f"Loading preprocessed data from {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    # Check if required columns exist
    required_columns = ['Content', 'Title', 'Category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Basic info about data
    logger.info(f"Loaded {len(df)} articles")
    logger.info(f"Number of categories: {df['Category'].nunique()}")
    logger.info(f"Categories with counts:")
    
    # Show top categories by count
    category_counts = df['Category'].value_counts()
    for category, count in category_counts.head(10).items():
        logger.info(f"  {category}: {count} articles")
    
    # Handle empty categories
    df['Category'] = df['Category'].fillna('Uncategorized')
    
    # Enhanced text preprocessing
    logger.info("Applying enhanced text preprocessing...")
    text_preprocessor = TextPreprocessor(
        language='czech',
        remove_stopwords=True,
        stemming=False,  # Vypneme stemming, protože nefunguje pro češtinu
        remove_accents=True,
        min_word_length=2
    )
    
    # Combine title and content for better classification
    df['Text'] = df['Title'] + ' ' + df['Content']
    
    # Apply preprocessing
    df['Processed_Text'] = df['Text'].progress_apply(text_preprocessor.preprocess_text)
    
    # Train Enhanced Category Classifier
    logger.info("Training Enhanced Category Classifier...")
    
    category_classifier_types = ['logistic_regression']
    category_results = {}
    
    for clf_type in category_classifier_types:
        logger.info(f"Training {clf_type} classifier for categories...")
        
        # Create classifier
        classifier = EnhancedCategoryClassifier(
            classifier_type=clf_type,
            min_samples_per_category=10,
            max_categories=15
        )
        
        # Train classifier
        results = classifier.fit(df['Processed_Text'], df['Category'], perform_grid_search=False)
        category_results[clf_type] = results
        
        # Save model
        model_dir = os.path.join(project_root, 'models', 'enhanced_category_classifier', clf_type)
        classifier.save_model(model_dir)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            results['class_names'],
            title=f'Category Confusion Matrix ({clf_type})'
        )
        
        # Save evaluation results
        results_path = os.path.join(
            project_root, 'reports', 'models', f'enhanced_category_classifier_{clf_type}_results.json'
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # Train Enhanced Sentiment Analyzer
    logger.info("Training Enhanced Sentiment Analyzer...")

    # Create artificial sentiment labels based on keywords
    logger.info("Creating artificial sentiment labels for training...")
    keywords = {
        'positive': ['úspěch', 'dobrý', 'skvělý', 'vynikající', 'radost', 'vítězství', 'zlepšení', 
                     'růst', 'zisk', 'podpora', 'inovace', 'výhra', 'schválení', 'oslava', 'pomoc'],
        'negative': ['problém', 'špatný', 'krize', 'smrt', 'neštěstí', 'katastrofa', 'nehoda',
                     'konflikt', 'válka', 'prohra', 'potíže', 'pokles', 'ztráta', 'obava', 'kritika']
    }

    # Function to label article based on keywords
    def assign_sentiment(text):
        text = text.lower()
        pos_count = sum(1 for kw in keywords['positive'] if kw in text)
        neg_count = sum(1 for kw in keywords['negative'] if kw in text)
        
        if pos_count > neg_count + 1:
            return 2  # positive
        elif neg_count > pos_count + 1:
            return 0  # negative
        else:
            return 1  # neutral

    # Assign sentiment to each article
    df['sentiment_label'] = df['Processed_Text'].apply(assign_sentiment)

    # Ensure we have at least some of each class
    sentiment_counts = df['sentiment_label'].value_counts()
    logger.info(f"Initial sentiment distribution: {sentiment_counts.to_dict()}")

    # If we're missing any class, manually assign some samples
    if 0 not in sentiment_counts or sentiment_counts[0] < 50:
        neutral_indices = df[df['sentiment_label'] == 1].index[:100].tolist()
        forced_negative = neutral_indices[:50]
        df.loc[forced_negative, 'sentiment_label'] = 0
        logger.info("Added forced negative samples")

    if 2 not in sentiment_counts or sentiment_counts[2] < 50:
        neutral_indices = df[df['sentiment_label'] == 1].index[100:200].tolist()
        forced_positive = neutral_indices[:50]
        df.loc[forced_positive, 'sentiment_label'] = 2
        logger.info("Added forced positive samples")

    # Verify distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    logger.info(f"Final sentiment distribution: {sentiment_counts.to_dict()}")

    sentiment_classifier_types = ['logistic_regression']
    sentiment_results = {}

    for clf_type in sentiment_classifier_types:
        logger.info(f"Training {clf_type} classifier for sentiment...")
        
        # Create sentiment analyzer
        analyzer = EnhancedSentimentAnalyzer(classifier_type=clf_type)
        
        # Train analyzer with our created labels
        results = analyzer.fit(df['Processed_Text'], y=df['sentiment_label'].values, perform_grid_search=False)
        sentiment_results[clf_type] = results
        
        # Save model
        model_dir = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer', clf_type)
        analyzer.save_model(model_dir)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            results['confusion_matrix'],
            analyzer.labels,
            title=f'Sentiment Confusion Matrix ({clf_type})'
        )
        
        # Plot class distribution
        plot_class_distribution(
            results['class_distribution'],
            analyzer.labels,
            title=f'Sentiment Distribution ({clf_type})'
        )
        
        # Save evaluation results
        results_path = os.path.join(
            project_root, 'reports', 'models', f'enhanced_sentiment_analyzer_{clf_type}_results.json'
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # Use only one classifier type for sentiment
    best_sentiment_clf = sentiment_classifier_types[0]
    best_sentiment_model_dir = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer', best_sentiment_clf)
    main_sentiment_model_dir = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')

    # Copy files from best model to main directory
    for filename in os.listdir(best_sentiment_model_dir):
        src = os.path.join(best_sentiment_model_dir, filename)
        dst = os.path.join(main_sentiment_model_dir, filename)
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)
    
    # Find best category classifier
    best_category_clf = list(category_results.keys())[0]
    logger.info(f"Best category classifier: {best_category_clf} with accuracy {category_results[best_category_clf]['accuracy']:.4f}")
    
    # Find best sentiment classifier
    best_sentiment_clf = list(sentiment_results.keys())[0]
    logger.info(f"Best sentiment classifier: {best_sentiment_clf} with accuracy {sentiment_results[best_sentiment_clf]['accuracy']:.4f}")
    
    # Copy best category model to main directory
    best_category_model_dir = os.path.join(project_root, 'models', 'enhanced_category_classifier', best_category_clf)
    main_category_model_dir = os.path.join(project_root, 'models', 'enhanced_category_classifier')
    
    # Copy files from best category model to main directory
    for filename in os.listdir(best_category_model_dir):
        src = os.path.join(best_category_model_dir, filename)
        dst = os.path.join(main_category_model_dir, filename)
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)
    
    logger.info(f"Training completed. Results saved to reports/models/")
    
    # Test models on a few sample articles to validate functionality
    logger.info("Testing models on sample articles...")
    
    # Load best models
    category_classifier = EnhancedCategoryClassifier.load_model(main_category_model_dir)
    sentiment_analyzer = EnhancedSentimentAnalyzer.load_model(main_sentiment_model_dir)
    
    # Sample some articles
    sample_idx = np.random.randint(0, len(df), size=5)
    samples = df.iloc[sample_idx]
    
    for _, article in samples.iterrows():
        # Category prediction
        predicted_category = category_classifier.predict([article['Processed_Text']])[0]
        
        # Sentiment prediction
        sentiment_id = sentiment_analyzer.predict([article['Processed_Text']])[0]
        sentiment = sentiment_analyzer.labels[sentiment_id]
        
        # Sentiment explanation
        explanation = sentiment_analyzer.explain_prediction(article['Processed_Text'])
        
        logger.info(f"Title: {article['Title'][:50]}...")
        logger.info(f"Category: {article['Category']} | Predicted: {predicted_category}")
        logger.info(f"Sentiment: {sentiment} | Reason: {explanation['reason']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    # Configure tqdm to work with pandas
    tqdm.pandas()
    
    try:
        main()
    except Exception as e:
        logger.exception("Unexpected error: %s", str(e))