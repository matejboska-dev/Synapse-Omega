import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_sentiment_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# absolute imports from src directory
from models.sentiment_analyzer import SentimentAnalyzer

# logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def plot_confusion_matrix(cm, class_names, figsize=(10, 8), cmap='Blues'):
    """
    plot confusion matrix
    
    args:
        cm (array): confusion matrix
        class_names (list): list of class names
        figsize (tuple): figure size
        cmap (str): colormap
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('reports/figures/sentiment_confusion_matrix.png')
    plt.close()

def plot_class_distribution(class_dist, class_names, figsize=(8, 6)):
    """
    plot class distribution
    
    args:
        class_dist (array): class distribution
        class_names (list): list of class names
        figsize (tuple): figure size
    """
    plt.figure(figsize=figsize)
    plt.bar(class_names, class_dist)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    plt.tight_layout()
    plt.savefig('reports/figures/sentiment_distribution.png')
    plt.close()

def main():
    """
    main function for training and evaluating the sentiment analyzer
    """
    # create output directories if they don't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for directory in ['models/sentiment_analyzer', 'reports/models']:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"ensuring directory exists: {dir_path}")
    
    # load preprocessed data
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'articles_processed.csv')
    
    if not os.path.exists(processed_data_path):
        logger.error(f"Preprocessed data file not found: {processed_data_path}")
        logger.error("Run data_preparation.py first")
        return
    
    logger.info(f"Loading preprocessed data from {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    # check if required columns exist
    required_columns = ['Content', 'Title']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # basic info about data
    logger.info(f"Loaded {len(df)} articles")
    
    # combine title and content for better analysis
    df['Text'] = df['Title'] + ' ' + df['Content']
    
    # train model with different classifier types
    classifiers = ['logistic_regression', 'svm', 'random_forest']
    results = {}
    
    for clf_type in classifiers:
        logger.info(f"Training {clf_type} classifier...")
        
        # create sentiment analyzer
        analyzer = SentimentAnalyzer(classifier_type=clf_type)
        
        # train analyzer (auto-generate labels since we don't have manual annotations)
        results[clf_type] = analyzer.fit(df['Text'])
        
        # save model
        model_dir = os.path.join(project_root, 'models', 'sentiment_analyzer', clf_type)
        analyzer.save_model(model_dir)
        
        # plot confusion matrix
        plot_confusion_matrix(
            results[clf_type]['confusion_matrix'],
            analyzer.labels,
            figsize=(8, 6)
        )
        
        # plot class distribution
        plot_class_distribution(
            results[clf_type]['class_distribution'],
            analyzer.labels,
            figsize=(8, 6)
        )
        
        # save evaluation results
        results_path = os.path.join(
            project_root, 'reports', 'models', f'sentiment_analyzer_{clf_type}_results.json'
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results[clf_type], f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # find best classifier
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"Best classifier: {best_clf[0]} with accuracy {best_clf[1]['accuracy']:.4f}")
    
    # create symbolic link or copy best model to main directory
    best_model_dir = os.path.join(project_root, 'models', 'sentiment_analyzer', best_clf[0])
    main_model_dir = os.path.join(project_root, 'models', 'sentiment_analyzer')
    
    # copy files from best model to main directory
    for filename in os.listdir(best_model_dir):
        src = os.path.join(best_model_dir, filename)
        dst = os.path.join(main_model_dir, filename)
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)
    
    logger.info(f"Training completed. Results saved to reports/models/")
    
    # test sentiment analyzer on a few sample articles
    logger.info("Testing sentiment analyzer on sample articles...")
    
    # load best model
    analyzer = SentimentAnalyzer.load_model(main_model_dir)
    
    # sample articles
    sample_idx = np.random.randint(0, len(df), size=5)
    samples = df.iloc[sample_idx]
    
    for _, article in samples.iterrows():
        sentiment = analyzer.predict([article['Text']])[0]
        logger.info(f"Title: {article['Title'][:50]}...")
        logger.info(f"Predicted sentiment: {analyzer.labels[sentiment]}")
        logger.info("-" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("unexpected error: %s", str(e))