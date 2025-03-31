import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_category_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# absolute imports from src directory
from models.category_classifier import CategoryClassifier

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
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figures_dir, 'category_confusion_matrix.png'))
    plt.close()

def plot_class_distribution(y, class_names, figsize=(12, 6)):
    """
    plot class distribution
    
    args:
        y (array): target labels
        class_names (list): list of class names
        figsize (tuple): figure size
    """
    plt.figure(figsize=figsize)
    counts = pd.Series(y).value_counts().reindex(class_names)
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Category Distribution')
    plt.tight_layout()
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figures_dir, 'category_distribution.png'))
    plt.close()

def preprocess_text(text):
    """
    Basic text preprocessing to improve classification
    
    args:
        text (str): input text
        
    returns:
        str: preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace common Czech accents
    text = text.replace('ě', 'e').replace('š', 's').replace('č', 'c')
    text = text.replace('ř', 'r').replace('ž', 'z').replace('ý', 'y')
    text = text.replace('á', 'a').replace('í', 'i').replace('é', 'e')
    text = text.replace('ú', 'u').replace('ů', 'u').replace('ň', 'n')
    text = text.replace('ť', 't').replace('ď', 'd')
    
    return text

def main():
    """
    main function for training and evaluating the category classifier
    """
    # create output directories if they don't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for directory in ['models/category_classifier', 'reports/models', 'reports/figures']:
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
    required_columns = ['Content', 'Title', 'Category']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # basic info about data
    logger.info(f"Loaded {len(df)} articles")
    logger.info(f"Number of categories: {df['Category'].nunique()}")
    logger.info(f"Categories with counts:")
    
    # show top categories by count
    category_counts = df['Category'].value_counts()
    for category, count in category_counts.head(10).items():
        logger.info(f"  {category}: {count} articles")
    
    # handle empty categories
    df['Category'] = df['Category'].fillna('Uncategorized')
    
    # Improve preprocessing with custom text cleaning
    df['Title_processed'] = df['Title'].fillna('').apply(preprocess_text)
    df['Content_processed'] = df['Content'].fillna('').apply(preprocess_text)
    
    # combine title and content for better classification (weighted title more heavily)
    df['Text'] = df['Title_processed'] + ' ' + df['Title_processed'] + ' ' + df['Content_processed']
    
    # train model with different classifier types
    classifiers = {
        'naive_bayes': {
            'min_samples': 10,
            'max_categories': 15
        },
        'logistic_regression': {
            'min_samples': 10,
            'max_categories': 15
        },
        'svm': {  # Added SVM classifier which often works better for text
            'min_samples': 10,
            'max_categories': 15
        }
    }
    
    results = {}
    
    for clf_type, params in classifiers.items():
        logger.info(f"Training {clf_type} classifier...")
        
        # create classifier
        classifier = CategoryClassifier(
            classifier_type=clf_type,
            min_samples_per_category=params['min_samples'],
            max_categories=params['max_categories']
        )
        
        # train classifier
        results[clf_type] = classifier.fit(df['Text'], df['Category'])
        
        # save model
        model_dir = os.path.join(project_root, 'models', 'category_classifier', clf_type)
        classifier.save_model(model_dir)
        
        # get predicted categories for the entire dataset to plot distribution
        categories = classifier.label_encoder.classes_
        
        # plot confusion matrix
        plot_confusion_matrix(
            results[clf_type]['confusion_matrix'],
            categories,
            figsize=(12, 10)
        )
        
        # save evaluation results
        results_path = os.path.join(
            project_root, 'reports', 'models', f'category_classifier_{clf_type}_results.json'
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results[clf_type], f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # find best classifier
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"Best classifier: {best_clf[0]} with accuracy {best_clf[1]['accuracy']:.4f}")
    
    # create symbolic link or copy best model to main directory
    best_model_dir = os.path.join(project_root, 'models', 'category_classifier', best_clf[0])
    main_model_dir = os.path.join(project_root, 'models', 'category_classifier')
    
    # copy files from best model to main directory
    for filename in os.listdir(best_model_dir):
        src = os.path.join(best_model_dir, filename)
        dst = os.path.join(main_model_dir, filename)
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)
    
    logger.info(f"Training completed. Results saved to reports/models/")
    
    # Test the model on a few examples
    logger.info("Testing the model on sample articles...")
    best_model = CategoryClassifier.load_model(main_model_dir)
    
    # Sample a few articles
    test_samples = df.sample(min(5, len(df)))
    for idx, row in test_samples.iterrows():
        predicted = best_model.predict([row['Text']])[0]
        logger.info(f"Article: {row['Title'][:50]}...")
        logger.info(f"True category: {row['Category']}, Predicted: {predicted}")
        logger.info('-' * 30)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("unexpected error: %s", str(e))