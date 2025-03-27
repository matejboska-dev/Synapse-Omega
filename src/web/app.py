import sys
import os
import pandas as pd
import json
import threading
import subprocess
from datetime import datetime
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# imports from our modules
from models.category_classifier import CategoryClassifier
from models.sentiment_analyzer import SentimentAnalyzer

# logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# create flask app
app = Flask(__name__)

# global variables
articles_df = None
category_model = None
sentiment_model = None
loaded_date = None

def run_scraper():
    """run scraper script in a separate process"""
    try:
        # get path to scraper script
        scraper_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'scraper.py')
        
        # execute scraper using the same python executable
        python_exe = sys.executable
        process = subprocess.Popen([python_exe, scraper_script], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
        
        logger.info("scraper started in background")
        
        # optionally wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("scraper completed successfully")
        else:
            logger.error(f"scraper failed with error: {stderr.decode('utf-8')}")
            
        # reload data after scraper completes
        load_data()
    except Exception as e:
        logger.error(f"error running scraper: {str(e)}")

def load_data():
    """load articles data and models"""
    global articles_df, category_model, sentiment_model, loaded_date
    
    # paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'processed', 'articles_processed.csv')
    category_model_path = os.path.join(project_root, 'models', 'category_classifier')
    sentiment_model_path = os.path.join(project_root, 'models', 'sentiment_analyzer')
    
    # load articles data
    try:
        articles_df = pd.read_csv(data_path)
        loaded_date = datetime.now()
        logger.info(f"loaded {len(articles_df)} articles from {data_path}")
    except Exception as e:
        logger.error(f"failed to load articles data: {str(e)}")
        articles_df = pd.DataFrame()
    
    # load category model if available
    try:
        category_model = CategoryClassifier.load_model(category_model_path)
        logger.info("category classifier loaded successfully")
    except Exception as e:
        logger.warning(f"could not load category model: {str(e)}")
        category_model = None
    
    # load sentiment model if available
    try:
        sentiment_model = SentimentAnalyzer.load_model(sentiment_model_path)
        logger.info("sentiment analyzer loaded successfully")
    except Exception as e:
        logger.warning(f"could not load sentiment model: {str(e)}")
        sentiment_model = None

@app.route('/')
def index():
    """home page showing article statistics and top categories"""
    # load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # get article stats
    stats = {
        'total_articles': len(articles_df) if articles_df is not None else 0,
        'sources': articles_df['Source'].nunique() if articles_df is not None else 0,
        'categories': articles_df['Category'].nunique() if articles_df is not None else 0,
        'date_range': {
            'from': articles_df['PublishDate'].min() if articles_df is not None else None,
            'to': articles_df['PublishDate'].max() if articles_df is not None else None
        },
        'newest_articles': articles_df.sort_values('PublishDate', ascending=False).head(5).to_dict('records') if articles_df is not None else [],
        'top_sources': articles_df['Source'].value_counts().head(5).to_dict() if articles_df is not None else {},
        'top_categories': articles_df['Category'].value_counts().head(5).to_dict() if articles_df is not None else {},
        'loaded_date': loaded_date
    }
    
    return render_template('index.html', stats=stats)

@app.route('/articles')
def articles():
    """page showing all articles with filters"""
    # load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # get filter parameters
    category = request.args.get('category', None)
    source = request.args.get('source', None)
    sentiment = request.args.get('sentiment', None)
    search = request.args.get('search', None)
    
    # apply filters
    filtered_df = articles_df.copy()
    
    if category and category != 'all':
        filtered_df = filtered_df[filtered_df['Category'] == category]
    
    if source and source != 'all':
        filtered_df = filtered_df[filtered_df['Source'] == source]
    
    if search:
        filtered_df = filtered_df[
            filtered_df['Title'].str.contains(search, case=False, na=False) | 
            filtered_df['Content'].str.contains(search, case=False, na=False)
        ]
    
    # pagination
    page = int(request.args.get('page', 1))
    per_page = 20
    total_pages = (len(filtered_df) + per_page - 1) // per_page
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    paged_df = filtered_df.sort_values('PublishDate', ascending=False).iloc[start_idx:end_idx]
    
    # prepare data for template
    articles_list = paged_df.to_dict('records')
    categories = articles_df['Category'].dropna().unique().tolist()
    sources = articles_df['Source'].dropna().unique().tolist()
    
    return render_template(
        'articles.html', 
        articles=articles_list,
        categories=sorted(categories),
        sources=sorted(sources),
        current_category=category,
        current_source=source,
        current_sentiment=sentiment,
        current_search=search,
        page=page,
        total_pages=total_pages,
        total_articles=len(filtered_df)
    )

@app.route('/article/<int:article_id>')
def article_detail(article_id):
    """page showing details of a specific article"""
    # load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # find article
    article = articles_df[articles_df['Id'] == article_id]
    
    if len(article) == 0:
        return render_template('error.html', message=f"Článek s ID {article_id} nebyl nalezen")
    
    article_data = article.iloc[0].to_dict()
    
    # get category prediction if model is available
    if category_model is not None:
        text = article_data['Title'] + ' ' + article_data['Content']
        predicted_category = category_model.predict([text])[0]
        article_data['predicted_category'] = predicted_category
    
    # get sentiment prediction if model is available
    if sentiment_model is not None:
        text = article_data['Title'] + ' ' + article_data['Content']
        sentiment_id = sentiment_model.predict([text])[0]
        article_data['sentiment'] = sentiment_model.labels[sentiment_id]
        
        # get sentiment features
        features = sentiment_model.extract_sentiment_features([text])
        article_data['sentiment_features'] = {
            'positive_word_count': features['positive_word_count'].iloc[0],
            'negative_word_count': features['negative_word_count'].iloc[0],
            'sentiment_ratio': features['sentiment_ratio'].iloc[0]
        }
    
    return render_template('article_detail.html', article=article_data)

@app.route('/categories')
def categories():
    """page showing article distribution across categories"""
    # load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # get category counts
    category_counts = articles_df['Category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    
    # prepare data for template
    categories_list = category_counts.to_dict('records')
    
    return render_template('categories.html', categories=categories_list)

@app.route('/sources')
def sources():
    """page showing article distribution across sources"""
    # load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # get source counts
    source_counts = articles_df['Source'].value_counts().reset_index()
    source_counts.columns = ['source', 'count']
    
    # prepare data for template
    sources_list = source_counts.to_dict('records')
    
    return render_template('sources.html', sources=sources_list)

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """API endpoint to analyze a text"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = {
        'text': text,
        'length': len(text),
        'word_count': len(text.split())
    }
    
    # get category prediction if model is available
    if category_model is not None:
        predicted_category = category_model.predict([text])[0]
        result['category'] = predicted_category
    
    # get sentiment prediction if model is available
    if sentiment_model is not None:
        sentiment_id = sentiment_model.predict([text])[0]
        result['sentiment'] = sentiment_model.labels[sentiment_id]
        
        # get sentiment features
        features = sentiment_model.extract_sentiment_features([text])
        result['sentiment_features'] = {
            'positive_word_count': int(features['positive_word_count'].iloc[0]),
            'negative_word_count': int(features['negative_word_count'].iloc[0]),
            'sentiment_ratio': float(features['sentiment_ratio'].iloc[0])
        }
    
    return jsonify(result)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """page with form to analyze custom text"""
    if request.method == 'POST':
        text = request.form.get('text', '')
        
        result = {
            'text': text,
            'length': len(text),
            'word_count': len(text.split())
        }
        
        # get category prediction if model is available
        if category_model is not None:
            predicted_category = category_model.predict([text])[0]
            result['category'] = predicted_category
        
        # get sentiment prediction if model is available
        if sentiment_model is not None:
            sentiment_id = sentiment_model.predict([text])[0]
            result['sentiment'] = sentiment_model.labels[sentiment_id]
            
            # get sentiment features
            features = sentiment_model.extract_sentiment_features([text])
            result['sentiment_features'] = {
                'positive_word_count': int(features['positive_word_count'].iloc[0]),
                'negative_word_count': int(features['negative_word_count'].iloc[0]),
                'sentiment_ratio': float(features['sentiment_ratio'].iloc[0])
            }
        
        return render_template('analyze.html', result=result)
    
    return render_template('analyze.html')

@app.route('/reload_data')
def reload_data():
    """endpoint to reload data and models"""
    # start scraper in background thread
    thread = threading.Thread(target=run_scraper)
    thread.daemon = True
    thread.start()
    
    # immediately reload data
    load_data()
    
    return redirect(url_for('index'))

@app.route('/run_scraper')
def trigger_scraper():
    """endpoint to manually trigger scraper"""
    # start scraper in background thread
    thread = threading.Thread(target=run_scraper)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "success", "message": "Scraper spuštěn na pozadí"})

@app.errorhandler(404)
def page_not_found(e):
    """handle 404 errors"""
    return render_template('error.html', message="Stránka nebyla nalezena"), 404

@app.errorhandler(500)
def server_error(e):
    """handle 500 errors"""
    return render_template('error.html', message="Chyba serveru"), 500

if __name__ == '__main__':
    # load data and models on startup
    load_data()
    
    # run scraper in background thread at startup
    thread = threading.Thread(target=run_scraper)
    thread.daemon = True
    thread.start()
    
    # run app in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)