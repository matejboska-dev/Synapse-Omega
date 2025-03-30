# src/scripts/train_simple_models.py
import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_simple_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
    """Plot confusion matrix"""
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
    """Plot class distribution"""
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

def main():
    """Main function for training and evaluating enhanced models"""
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
    
    # Basic info about data
    logger.info(f"Loaded {len(df)} articles")
    logger.info(f"Number of categories: {df['Category'].nunique()}")
    
    # Handle empty categories
    df['Category'] = df['Category'].fillna('Uncategorized')
    
    # Apply simple preprocessing
    logger.info("Applying simple text preprocessing...")
    df['Processed_Text'] = df['Title'] + ' ' + df['Content']
    
    # Train Category Classifier (without grid search)
    logger.info("Training Enhanced Category Classifier...")
    classifier_type = 'logistic_regression'  # Nejstabilnější klasifikátor
    
    # Create classifier
    classifier = EnhancedCategoryClassifier(
        classifier_type=classifier_type,
        min_samples_per_category=10,
        max_categories=15
    )
    
    # Train classifier without grid search
    results = classifier.fit(df['Processed_Text'], df['Category'], perform_grid_search=False)
    
    # Save model
    model_dir = os.path.join(project_root, 'models', 'enhanced_category_classifier')
    classifier.save_model(model_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        results['class_names'],
        title=f'Category Confusion Matrix'
    )
    
    # Save evaluation results
    results_path = os.path.join(
        project_root, 'reports', 'models', f'enhanced_category_classifier_results.json'
    )
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # Train Sentiment Analyzer (without grid search)
    logger.info("Training Enhanced Sentiment Analyzer...")
    
    # Create sentiment analyzer
    analyzer = EnhancedSentimentAnalyzer(classifier_type=classifier_type)
    
    # Train analyzer
    results = analyzer.fit(df['Processed_Text'], perform_grid_search=False)
    
    # Save model
    model_dir = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')
    analyzer.save_model(model_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        analyzer.labels,
        title=f'Sentiment Confusion Matrix'
    )
    
    # Plot class distribution
    plot_class_distribution(
        results['class_distribution'],
        analyzer.labels,
        title=f'Sentiment Distribution'
    )
    
    # Save evaluation results
    results_path = os.path.join(
        project_root, 'reports', 'models', f'enhanced_sentiment_analyzer_results.json'
    )
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Training completed. Results saved to reports/models/")
    
    # Test models on a few sample articles
    logger.info("Testing models on sample articles...")
    
    # Sample some articles
    sample_idx = np.random.randint(0, len(df), size=5)
    samples = df.iloc[sample_idx]
    
    for _, article in samples.iterrows():
        # Category prediction
        predicted_category = classifier.predict([article['Processed_Text']])[0]
        
        # Sentiment prediction
        sentiment_id = analyzer.predict([article['Processed_Text']])[0]
        sentiment = analyzer.labels[sentiment_id]
        
        # Get explanation
        explanation = analyzer.explain_prediction(article['Processed_Text'])
        
        logger.info(f"Title: {article['Title'][:50]}...")
        logger.info(f"Category: {article['Category']} | Predicted: {predicted_category}")
        logger.info(f"Sentiment: {sentiment} | Reason: {explanation['reason']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unexpected error: %s", str(e))