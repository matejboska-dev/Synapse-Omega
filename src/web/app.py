import sys
import os
import pandas as pd
import json
import threading
import subprocess
from datetime import datetime
import logging
import glob
import pyodbc
from flask import Flask, render_template, request, redirect, url_for, jsonify
from routes.chatbot import article_chatbot_api

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# imports from our modules
from models.category_classifier import CategoryClassifier
from models.sentiment_analyzer import SentimentAnalyzer
from models.enhanced_category_classifier import EnhancedCategoryClassifier
from models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from routes.chatbot import chatbot_bp

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

# register blueprints
app.register_blueprint(chatbot_bp)
app.add_url_rule('/api/article_chatbot', view_func=article_chatbot_api, methods=['POST'])

# global variables
articles_df = None
category_model = None
sentiment_model = None
loaded_date = None
enhanced_models = False

def run_daily_scraper():
    """Run scraper script to collect the latest news articles"""
    try:
        # Check if required dependencies are installed
        try:
            import feedparser
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            logger.error(f"Chybí potřebná závislost pro scraper: {str(e)}")
            logger.error("Nainstalujte potřebné balíčky: pip install feedparser requests beautifulsoup4 pyodbc tqdm")
            return
            
        # get path to scraper script
        scraper_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'scraper.py')
        
        # execute scraper using the same python executable
        python_exe = sys.executable
        
        # Run with limited article count (just latest headlines - 3 per source)
        # Add timeout to prevent hanging
        process = subprocess.Popen(
            [python_exe, scraper_script, '--latest', '--max-per-source=3'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        logger.info("Latest news scraper started - sbírám nejnovější zprávy")
        
        # Wait for completion with timeout
        try:
            stdout, stderr = process.communicate(timeout=60)  # 60 seconds timeout
            
            if process.returncode == 0:
                logger.info("Latest news scraper completed successfully")
            else:
                logger.error(f"Latest news scraper failed with error: {stderr.decode('utf-8')}")
        except subprocess.TimeoutExpired:
            process.kill()
            logger.warning("Latest news scraper timed out after 60 seconds, process killed")
            
        # reload data after scraper completes
        load_data()
    except Exception as e:
        logger.error(f"Error running latest news scraper: {str(e)}")

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
    """load only scraped articles data (not training data) and models"""
    global articles_df, category_model, sentiment_model, loaded_date, enhanced_models
    
    # paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    display_dir = os.path.join(project_root, 'data', 'display')
    display_file = os.path.join(display_dir, 'all_articles.json')
    
    # Check for enhanced models first
    enhanced_category_model_path = os.path.join(project_root, 'models', 'enhanced_category_classifier')
    enhanced_sentiment_model_path = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')
    
    standard_category_model_path = os.path.join(project_root, 'models', 'category_classifier')
    standard_sentiment_model_path = os.path.join(project_root, 'models', 'sentiment_analyzer')
    
    # create display directory if it doesn't exist
    os.makedirs(display_dir, exist_ok=True)
    
    # load articles data
    try:
        # Try to load from local files first
        local_data_loaded = False
        
        # Check if all_articles.json exists first
        if os.path.exists(display_file):
            try:
                with open(display_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                articles_df = pd.DataFrame(articles)
                logger.info(f"Loaded {len(articles_df)} articles from {display_file}")
                local_data_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load from all_articles.json: {str(e)}")
        
        # If local file loading failed, try database
        if not local_data_loaded:
            try:
                # Connection parameters
                server = "193.85.203.188"
                database = "boska"
                username = "boska"
                password = "123456"
                
                # Try to load configuration
                config_path = os.path.join(project_root, 'config', 'database.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        server = config.get('server', server)
                        database = config.get('database', database)
                        username = config.get('username', username)
                        password = config.get('password', password)
                    logger.info("Configuration loaded from config/database.json")
                
                # Check if pyodbc is available
                try:
                    import pyodbc
                except ImportError:
                    logger.warning("pyodbc module not available, cannot connect to database")
                    raise ImportError("pyodbc not installed")
                
                # Attempt database connection
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
                conn = pyodbc.connect(conn_str)
                
                # Load articles directly from database
                query = """
                SELECT Id, SourceName as Source, Title, ArticleUrl, PublicationDate as PublishDate, 
                      Category, ArticleLength, WordCount, ArticleText as Content, 
                      ScrapedDate
                FROM Articles
                """
                articles_df = pd.read_sql(query, conn)
                conn.close()
                
                # Save the loaded data to display files for next time
                os.makedirs(display_dir, exist_ok=True)
                articles_df.to_json(display_file, orient='records', force_ascii=False, indent=2)
                
                logger.info(f"Loaded {len(articles_df)} articles directly from database")
                
            except Exception as db_error:
                logger.warning(f"Could not load from database: {str(db_error)}, trying other local files")
                
                # database connection failed, try other local files
                # find latest display file
                display_files = glob.glob(os.path.join(display_dir, 'display_articles_*.json'))
                
                if display_files:
                    latest_file = max(display_files, key=os.path.getmtime)
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    articles_df = pd.DataFrame(articles)
                    logger.info(f"Loaded {len(articles_df)} articles from {latest_file}")
                else:
                    # Try looking in data/processed_scraped directory
                    processed_dir = os.path.join(project_root, 'data', 'processed_scraped')
                    processed_files = glob.glob(os.path.join(processed_dir, 'processed_articles_*.json'))
                    
                    if processed_files:
                        latest_processed = max(processed_files, key=os.path.getmtime)
                        with open(latest_processed, 'r', encoding='utf-8') as f:
                            articles = json.load(f)
                        articles_df = pd.DataFrame(articles)
                        logger.info(f"Loaded {len(articles_df)} articles from {latest_processed}")
                    else:
                        # Try looking in data/scraped directory
                        scraped_dir = os.path.join(project_root, 'data', 'scraped')
                        os.makedirs(scraped_dir, exist_ok=True)
                        scraped_files = glob.glob(os.path.join(scraped_dir, 'articles_*.json'))
                        
                        if scraped_files:
                            latest_scraped = max(scraped_files, key=os.path.getmtime)
                            with open(latest_scraped, 'r', encoding='utf-8') as f:
                                articles = json.load(f)
                            articles_df = pd.DataFrame(articles)
                            logger.info(f"Loaded {len(articles_df)} articles from {latest_scraped}")
                        else:
                            # if no display files found, create empty DataFrame with required columns
                            articles_df = pd.DataFrame(columns=[
                                'Id', 'Title', 'Content', 'Source', 'Category', 'predicted_category', 
                                'sentiment', 'PublishDate', 'ArticleUrl', 'ArticleLength', 'WordCount'
                            ])
                            logger.warning("No article data files found, created empty DataFrame")
        
        # Ensure Id column exists and is unique
        if 'Id' not in articles_df.columns:
            articles_df['Id'] = range(1, len(articles_df) + 1)
        
        # Make sure all required columns exist
        for col in ['Title', 'Content', 'Source', 'Category', 'PublishDate', 'ArticleLength', 'WordCount']:
            if col not in articles_df.columns:
                if col == 'Source' and 'SourceName' in articles_df.columns:
                    articles_df['Source'] = articles_df['SourceName']
                elif col == 'Content' and 'ArticleText' in articles_df.columns:
                    articles_df['Content'] = articles_df['ArticleText']
                elif col == 'PublishDate' and 'PublicationDate' in articles_df.columns:
                    articles_df['PublishDate'] = articles_df['PublicationDate']
                else:
                    articles_df[col] = None
        
        loaded_date = datetime.now()
        app.config['articles_df'] = articles_df
        
    except Exception as e:
        logger.error(f"Failed to load articles data: {str(e)}")
        articles_df = pd.DataFrame(columns=[
            'Id', 'Title', 'Content', 'Source', 'Category', 'predicted_category', 
            'sentiment', 'PublishDate', 'ArticleUrl', 'ArticleLength', 'WordCount'
        ])
        app.config['articles_df'] = articles_df
    
    # Load models
    try:
        # Try to load enhanced category model first
        if os.path.exists(enhanced_category_model_path):
            category_model = EnhancedCategoryClassifier.load_model(enhanced_category_model_path)
            logger.info("Enhanced category classifier loaded successfully")
            enhanced_models = True
        else:
            # Fallback to standard model
            try:
                category_model = CategoryClassifier.load_model(standard_category_model_path)
                logger.info("Standard category classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load standard category model: {str(e)}")
                category_model = None
            
        # Try to load enhanced sentiment model first
        if os.path.exists(enhanced_sentiment_model_path):
            sentiment_model = EnhancedSentimentAnalyzer.load_model(enhanced_sentiment_model_path)
            logger.info("Enhanced sentiment analyzer loaded successfully")
            enhanced_models = True
        else:
            # Fallback to standard model
            try:
                sentiment_model = SentimentAnalyzer.load_model(standard_sentiment_model_path)
                logger.info("Standard sentiment analyzer loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load standard sentiment model: {str(e)}")
                sentiment_model = None
    except Exception as e:
        logger.warning(f"Could not load models: {str(e)}")
        enhanced_models = False
        category_model = None
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
        'sources': articles_df['Source'].nunique() if articles_df is not None and len(articles_df) > 0 else 0,
        'categories': articles_df['Category'].nunique() if articles_df is not None and len(articles_df) > 0 else 0,
        'date_range': {
            'from': articles_df['PublishDate'].min() if articles_df is not None and len(articles_df) > 0 else None,
            'to': articles_df['PublishDate'].max() if articles_df is not None and len(articles_df) > 0 else None
        },
        'newest_articles': articles_df.sort_values('PublishDate', ascending=False).head(5).to_dict('records') if articles_df is not None and len(articles_df) > 0 else [],
        'top_sources': articles_df['Source'].value_counts().head(5).to_dict() if articles_df is not None and len(articles_df) > 0 else {},
        'top_categories': articles_df['Category'].value_counts().head(5).to_dict() if articles_df is not None and len(articles_df) > 0 else {},
        'loaded_date': loaded_date,
        'enhanced_models': enhanced_models
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
    if len(articles_df) > 0:
        filtered_df = articles_df.copy()
        
        if category and category != 'all':
            filtered_df = filtered_df[filtered_df['Category'] == category]
        
        if source and source != 'all':
            filtered_df = filtered_df[filtered_df['Source'] == source]
        
        if sentiment and sentiment != 'all':
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
        
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
        sentiments = articles_df['sentiment'].dropna().unique().tolist() if 'sentiment' in articles_df.columns else []
    else:
        # empty data
        articles_list = []
        categories = []
        sources = []
        sentiments = []
        page = 1
        total_pages = 0
        filtered_df = pd.DataFrame()
    
    return render_template(
        'articles.html', 
        articles=articles_list,
        categories=sorted(categories),
        sources=sorted(sources),
        sentiments=sorted(sentiments) if sentiments else [],
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
    
    return render_template('article_detail.html', article=article_data)

@app.route('/categories')
def categories():
    """page showing article distribution across categories"""
    # load data if not loaded yet
    if articles_df is None:
        load_data()
    
    # get category counts
    if len(articles_df) > 0:
        category_counts = articles_df['Category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
    else:
        category_counts = pd.DataFrame(columns=['category', 'count'])
    
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
    if len(articles_df) > 0:
        source_counts = articles_df['Source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
    else:
        source_counts = pd.DataFrame(columns=['source', 'count'])
    
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
        # Handle different model types
        if enhanced_models:
            # Enhanced models have different processing requirements
            from data.text_preprocessor import TextPreprocessor
            text_preprocessor = TextPreprocessor(language='czech')
            processed_text = text_preprocessor.preprocess_text(text)
            
            sentiment_id = sentiment_model.predict([processed_text])[0]
            result['sentiment'] = sentiment_model.labels[sentiment_id]
            
            # Add explanation for enhanced models
            explanation = sentiment_model.explain_prediction(processed_text)
            result['sentiment_features'] = {
                'positive_word_count': int(explanation['positive_word_count']),
                'negative_word_count': int(explanation['negative_word_count']),
                'sentiment_ratio': float(explanation['sentiment_ratio']),
                'positive_words': explanation['positive_words'],
                'negative_words': explanation['negative_words'],
                'reason': explanation['reason']
            }
        else:
            # Standard model processing
            sentiment_id = sentiment_model.predict([text])[0]
            result['sentiment'] = sentiment_model.labels[sentiment_id]
            
            # Get sentiment features
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
            # Handle different model types
            if enhanced_models:
                from data.text_preprocessor import TextPreprocessor
                text_preprocessor = TextPreprocessor(language='czech')
                processed_text = text_preprocessor.preprocess_text(text)
                predicted_category = category_model.predict([processed_text])[0]
            else:
                predicted_category = category_model.predict([text])[0]
                
            result['category'] = predicted_category
        
        # get sentiment prediction if model is available
        if sentiment_model is not None:
            # Handle different model types
            if enhanced_models:
                # Enhanced models have different processing requirements
                from data.text_preprocessor import TextPreprocessor
                text_preprocessor = TextPreprocessor(language='czech')
                processed_text = text_preprocessor.preprocess_text(text)
                
                sentiment_id = sentiment_model.predict([processed_text])[0]
                result['sentiment'] = sentiment_model.labels[sentiment_id]
                
                # Add explanation for enhanced models
                explanation = sentiment_model.explain_prediction(processed_text)
                result['sentiment_features'] = {
                    'positive_word_count': int(explanation['positive_word_count']),
                    'negative_word_count': int(explanation['negative_word_count']),
                    'sentiment_ratio': float(explanation['sentiment_ratio']),
                    'positive_words': explanation['positive_words'],
                    'negative_words': explanation['negative_words'],
                    'reason': explanation['reason']
                }
            else:
                # Standard model processing
                sentiment_id = sentiment_model.predict([text])[0]
                result['sentiment'] = sentiment_model.labels[sentiment_id]
                
                # Get sentiment features
                features = sentiment_model.extract_sentiment_features([text])
                result['sentiment_features'] = {
                    'positive_word_count': int(features['positive_word_count'].iloc[0]),
                    'negative_word_count': int(features['negative_word_count'].iloc[0]),
                    'sentiment_ratio': float(features['sentiment_ratio'].iloc[0])
                }
        
        return render_template('analyze.html', result=result, enhanced_models=enhanced_models)
    
    return render_template('analyze.html', enhanced_models=enhanced_models)

@app.route('/reload_data')
def reload_data():
    """endpoint to reload data and models"""
    # start latest news scraper in background thread
    thread = threading.Thread(target=run_daily_scraper)
    thread.daemon = True
    thread.start()
    
    return redirect(url_for('index'))

@app.route('/run_scraper')
def trigger_scraper():
    """endpoint to manually trigger scraper"""
    # start scraper in background thread
    thread = threading.Thread(target=run_scraper)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "success", "message": "Scraper spuštěn na pozadí"})

@app.route('/train_enhanced_models')
def train_enhanced_models():
    """endpoint to train enhanced models"""
    try:
        # get path to training script
        train_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'train_enhanced_models.py')
        
        if not os.path.exists(train_script):
            return jsonify({"status": "error", "message": "Script pro trénování vylepšených modelů nebyl nalezen."}), 404
        
        # execute training script using the same python executable in a separate process
        python_exe = sys.executable
        process = subprocess.Popen([python_exe, train_script],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        
        logger.info("Enhanced model training started in background")
        
        # Return immediately, don't wait for completion
        return jsonify({"status": "success", "message": "Trénování vylepšených modelů spuštěno na pozadí."})
        
    except Exception as e:
        logger.error(f"Error starting enhanced model training: {str(e)}")
        return jsonify({"status": "error", "message": f"Chyba při spouštění trénování: {str(e)}"}), 500

@app.errorhandler(404)
def page_not_found(e):
    """handle 404 errors"""
    return render_template('error.html', message="Stránka nebyla nalezena"), 404

@app.errorhandler(500)
def server_error(e):
    """handle 500 errors"""
    return render_template('error.html', message="Chyba serveru"), 500

if __name__ == '__main__':
    # Initialize database connection if needed
    try:
        # Check if database connection can be established
        from data.database_connector import DatabaseConnector
        
        # Get database connection parameters - from config if available
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, 'config', 'database.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                server = config.get('server', "193.85.203.188")
                database = config.get('database', "boska")
                username = config.get('username', "boska")
                password = config.get('password', "123456")
        else:
            # Default connection parameters
            server = "193.85.203.188"
            database = "boska"
            username = "boska"
            password = "123456"
        
        # Test connection
        db_connector = DatabaseConnector(server, database, username, password)
        if db_connector.connect():
            logger.info("Database connection successful - ready for scraper")
            db_connector.disconnect()
        else:
            logger.warning("Could not connect to database - scraper will save data locally")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    
    # Run scraper to get latest news (5 per source) before starting the app
    try:
        logger.info("Running initial scraper to collect latest news (5 per source)...")
        # Using subprocess.run with check=True to ensure it completes or raises exception
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'scripts', 'scraper.py'), '--latest', '--max-per-source=5'],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Initial scraping completed successfully: {result.stdout}")
        
        # Process the scraped data
        logger.info("Processing scraped data...")
        process_result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'scripts', 'process_scraped_data.py')],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Data processing completed: {process_result.stdout}")
        
        # Load data after scraping
        load_data()
    except subprocess.CalledProcessError as e:
        logger.error(f"Scraper or processing failed with error: {e.stderr}")
    except Exception as e:
        logger.error(f"Failed to run initial scraper: {str(e)}")
    
    # start the app
    app.run(debug=True, host='0.0.0.0', port=5000)
    # load data and models on startup
    load_data()
    
    # run daily scraper in background thread at startup
    # Wrap in try-except to prevent app from failing if scraper fails
    try:
        daily_thread = threading.Thread(target=run_daily_scraper)
        daily_thread.daemon = True
        daily_thread.start()
        logger.info("Daily scraper thread started")
    except Exception as e:
        logger.error(f"Failed to start scraper thread: {str(e)}")
    
    # run app in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)