import sys
import os
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
import random
import json
import pyodbc
from urllib.parse import urlparse

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ensure data directory exists
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'scraped')
os.makedirs(data_dir, exist_ok=True)

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

class NewsScraper:
    """
    scraper for czech news websites
    """
    
    def __init__(self, db_config=None):
        """
        initialize the scraper
        
        args:
            db_config (dict): database configuration
        """
        self.db_config = db_config or {
            'server': '193.85.203.188',
            'database': 'boska',
            'username': 'boska',
            'password': '123456',
            'driver': '{ODBC Driver 17 for SQL Server}'
        }
        self.conn = None
        self.cursor = None
        
        # user agent to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # today's date for saving scraped data
        self.today = datetime.now().strftime('%Y-%m-%d')
        
        # sources configuration
        self.sources = [
            {
                'name': 'idnes',
                'rss_url': 'https://servis.idnes.cz/rss.aspx?c=zpravodaj',
                'parser': self.parse_idnes
            },
            {
                'name': 'novinky',
                'rss_url': 'https://www.novinky.cz/rss',
                'parser': self.parse_novinky
            },
            {
                'name': 'irozhlas',
                'rss_url': 'https://www.irozhlas.cz/rss/irozhlas',
                'parser': self.parse_irozhlas
            },
            {
                'name': 'seznamzpravy',
                'rss_url': 'https://www.seznamzpravy.cz/rss',
                'parser': self.parse_seznamzpravy
            },
            {
                'name': 'aktualne',
                'rss_url': 'https://www.aktualne.cz/rss/',
                'parser': self.parse_aktualne
            }
        ]
        
        # initialize scraped articles container
        self.scraped_articles = []
    
    def connect_to_db(self):
        """
        connect to the database
        
        returns:
            bool: whether connection was successful
        """
        try:
            conn_str = f"DRIVER={self.db_config['driver']};SERVER={self.db_config['server']};DATABASE={self.db_config['database']};UID={self.db_config['username']};PWD={self.db_config['password']}"
            self.conn = pyodbc.connect(conn_str)
            self.cursor = self.conn.cursor()
            logger.info("successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"error connecting to database: {str(e)}")
            return False
    
    def disconnect_from_db(self):
        """disconnect from the database"""
        if self.conn:
            self.conn.close()
            logger.info("disconnected from database")
    
    def get_existing_urls(self):
        """
        get list of article urls already in the database
        
        returns:
            list: list of existing urls
        """
        if not self.conn:
            if not self.connect_to_db():
                return []
        
        try:
            self.cursor.execute("SELECT ArticleUrl FROM Articles")
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"error fetching existing urls: {str(e)}")
            return []
    
    def save_article(self, article):
        """
        save article to database and to local scraped data
        
        args:
            article (dict): article data
            
        returns:
            bool: whether save was successful
        """
        # add to scraped articles list
        self.scraped_articles.append(article)
        
        if not self.conn:
            if not self.connect_to_db():
                return False
        
        try:
            # check if article already exists
            self.cursor.execute("SELECT COUNT(*) FROM Articles WHERE ArticleUrl = ?", article['ArticleUrl'])
            if self.cursor.fetchone()[0] > 0:
                logger.info(f"article already exists: {article['Title']}")
                return False
                
            # insert article
            query = """
            INSERT INTO Articles 
            (Source, Title, ArticleUrl, PublishDate, Category, ArticleLength, WordCount, Content, ScrapedDate) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.cursor.execute(
                query,
                article['Source'],
                article['Title'],
                article['ArticleUrl'],
                article['PublishDate'],
                article['Category'],
                article['ArticleLength'],
                article['WordCount'],
                article['Content'],
                datetime.now()
            )
            
            self.conn.commit()
            logger.info(f"saved article: {article['Title']}")
            return True
        except Exception as e:
            logger.error(f"error saving article: {str(e)}")
            return False
    
    def save_to_file(self):
        """
        save scraped articles to a json file
        """
        if not self.scraped_articles:
            logger.info("no new articles to save to file")
            return
            
        # create filename with date
        filename = f"articles_{self.today}.json"
        filepath = os.path.join(data_dir, filename)
        
        try:
            # convert datetime objects to strings for json serialization
            articles_to_save = []
            for article in self.scraped_articles:
                article_copy = article.copy()
                if isinstance(article_copy.get('PublishDate'), datetime):
                    article_copy['PublishDate'] = article_copy['PublishDate'].strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(article_copy.get('ScrapedDate'), datetime):
                    article_copy['ScrapedDate'] = article_copy['ScrapedDate'].strftime('%Y-%m-%d %H:%M:%S')
                articles_to_save.append(article_copy)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(articles_to_save, f, ensure_ascii=False, indent=2)
                
            logger.info(f"saved {len(self.scraped_articles)} articles to {filepath}")
        except Exception as e:
            logger.error(f"error saving articles to file: {str(e)}")
    
    def fetch_rss_feed(self, url):
        """
        fetch and parse rss feed
        
        args:
            url (str): rss feed url
            
        returns:
            list: list of dictionaries with article data
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"failed to fetch rss feed: {url}, status code: {response.status_code}")
                return []
                
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            articles = []
            for item in items:
                try:
                    article = {
                        'Title': item.title.text if item.title else '',
                        'ArticleUrl': item.link.text if item.link else '',
                        'PublishDate': item.pubDate.text if item.pubDate else '',
                        'Category': item.category.text if hasattr(item, 'category') and item.category else '',
                        'Description': item.description.text if item.description else ''
                    }
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"error parsing rss item: {str(e)}")
                    continue
                
            logger.info(f"fetched {len(articles)} articles from {url}")
            return articles
        except Exception as e:
            logger.error(f"error fetching rss feed: {url}, error: {str(e)}")
            return []
    
    def parse_idnes(self, article_url):
        """
        parse article from idnes.cz
        
        args:
            article_url (str): url of the article
            
        returns:
            dict: article data or None if parsing failed
        """
        try:
            response = requests.get(article_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # title
            title = soup.find('h1', class_='article-title')
            if not title:
                title = soup.find('h1')
            title = title.text.strip() if title else ''
            
            # content
            content_div = soup.find('div', class_='article-content')
            if not content_div:
                content_div = soup.find('div', class_='article-body')
                
            if content_div:
                # remove unwanted elements
                for unwanted in content_div.find_all(['script', 'style', 'aside', 'figure', 'figcaption']):
                    unwanted.decompose()
                    
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs])
            else:
                content = ''
            
            # category
            category = ''
            category_elem = soup.find('span', class_='breadcrumbs__item')
            if category_elem:
                category = category_elem.text.strip()
            
            # date
            date = soup.find('div', class_='article-info')
            if date:
                date = date.find('span', class_='article-info__date')
                date = date.text.strip() if date else ''
            else:
                date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'Title': title,
                'Content': content,
                'Category': category,
                'PublishDate': date,
                'ArticleLength': len(content),
                'WordCount': len(content.split()),
                'Source': 'idnes',
                'ArticleUrl': article_url,
                'ScrapedDate': datetime.now()
            }
        except Exception as e:
            logger.error(f"error parsing idnes article: {article_url}, error: {str(e)}")
            return None
    
    def parse_novinky(self, article_url):
        """
        parse article from novinky.cz
        
        args:
            article_url (str): url of the article
            
        returns:
            dict: article data or None if parsing failed
        """
        try:
            response = requests.get(article_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # title
            title = soup.find('h1', class_='article-title')
            if not title:
                title = soup.find('h1')
            title = title.text.strip() if title else ''
            
            # content
            content_div = soup.find('div', class_='article-content')
            if not content_div:
                content_div = soup.find('div', class_='article__text')
                
            if content_div:
                # remove unwanted elements
                for unwanted in content_div.find_all(['script', 'style', 'aside', 'figure', 'figcaption']):
                    unwanted.decompose()
                    
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs])
            else:
                content = ''
            
            # category
            category = ''
            category_elem = soup.find('span', class_='breadcrumb__item--active')
            if category_elem:
                category = category_elem.text.strip()
            
            # date
            date = soup.find('span', class_='article-info__date')
            if date:
                date = date.text.strip()
            else:
                date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'Title': title,
                'Content': content,
                'Category': category,
                'PublishDate': date,
                'ArticleLength': len(content),
                'WordCount': len(content.split()),
                'Source': 'novinky',
                'ArticleUrl': article_url,
                'ScrapedDate': datetime.now()
            }
        except Exception as e:
            logger.error(f"error parsing novinky article: {article_url}, error: {str(e)}")
            return None
    
    def parse_irozhlas(self, article_url):
        """
        parse article from irozhlas.cz
        
        args:
            article_url (str): url of the article
            
        returns:
            dict: article data or None if parsing failed
        """
        try:
            response = requests.get(article_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # title
            title = soup.find('h1', class_='article-title')
            if not title:
                title = soup.find('h1')
            title = title.text.strip() if title else ''
            
            # content
            content_div = soup.find('div', class_='article-body')
            if not content_div:
                content_div = soup.find('div', class_='b-detail')
                
            if content_div:
                # remove unwanted elements
                for unwanted in content_div.find_all(['script', 'style', 'aside', 'figure', 'figcaption']):
                    unwanted.decompose()
                    
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs])
            else:
                content = ''
            
            # category
            category = ''
            category_elem = soup.find('span', class_='breadcrumb__item--active')
            if category_elem:
                category = category_elem.text.strip()
            
            # date
            date = soup.find('span', class_='article-info__date')
            if date:
                date = date.text.strip()
            else:
                date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'Title': title,
                'Content': content,
                'Category': category,
                'PublishDate': date,
                'ArticleLength': len(content),
                'WordCount': len(content.split()),
                'Source': 'irozhlas',
                'ArticleUrl': article_url,
                'ScrapedDate': datetime.now()
            }
        except Exception as e:
            logger.error(f"error parsing irozhlas article: {article_url}, error: {str(e)}")
            return None
    
    def parse_seznamzpravy(self, article_url):
        """
        parse article from seznamzpravy.cz
        
        args:
            article_url (str): url of the article
            
        returns:
            dict: article data or None if parsing failed
        """
        try:
            response = requests.get(article_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # title
            title = soup.find('h1', class_='article-title')
            if not title:
                title = soup.find('h1')
            title = title.text.strip() if title else ''
            
            # content
            content_div = soup.find('div', class_='article-content')
            if not content_div:
                content_div = soup.find('div', class_='szn-article__text')
                
            if content_div:
                # remove unwanted elements
                for unwanted in content_div.find_all(['script', 'style', 'aside', 'figure', 'figcaption']):
                    unwanted.decompose()
                    
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs])
            else:
                content = ''
            
            # category
            category = ''
            category_elem = soup.find('span', class_='breadcrumb__item--active')
            if category_elem:
                category = category_elem.text.strip()
            
            # date
            date = soup.find('span', class_='article-info__date')
            if date:
                date = date.text.strip()
            else:
                date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'Title': title,
                'Content': content,
                'Category': category,
                'PublishDate': date,
                'ArticleLength': len(content),
                'WordCount': len(content.split()),
                'Source': 'seznamzpravy',
                'ArticleUrl': article_url,
                'ScrapedDate': datetime.now()
            }
        except Exception as e:
            logger.error(f"error parsing seznamzpravy article: {article_url}, error: {str(e)}")
            return None
    
    def parse_aktualne(self, article_url):
        """
        parse article from aktualne.cz
        
        args:
            article_url (str): url of the article
            
        returns:
            dict: article data or None if parsing failed
        """
        try:
            response = requests.get(article_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # title
            title = soup.find('h1', class_='article-title')
            if not title:
                title = soup.find('h1')
            title = title.text.strip() if title else ''
            
            # content
            content_div = soup.find('div', class_='article-content')
            if not content_div:
                content_div = soup.find('div', class_='text')
                
            if content_div:
                # remove unwanted elements
                for unwanted in content_div.find_all(['script', 'style', 'aside', 'figure', 'figcaption']):
                    unwanted.decompose()
                    
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs])
            else:
                content = ''
            
            # category
            category = ''
            category_elem = soup.find('div', class_='breadcrumb')
            if category_elem:
                category_links = category_elem.find_all('a')
                if category_links and len(category_links) > 0:
                    category = category_links[-1].text.strip()
            
            # date
            date = soup.find('span', class_='date')
            if date:
                date = date.text.strip()
            else:
                date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'Title': title,
                'Content': content,
                'Category': category,
                'PublishDate': date,
                'ArticleLength': len(content),
                'WordCount': len(content.split()),
                'Source': 'aktualne',
                'ArticleUrl': article_url,
                'ScrapedDate': datetime.now()
            }
        except Exception as e:
            logger.error(f"error parsing aktualne article: {article_url}, error: {str(e)}")
            return None
    
    def get_source_parser(self, url):
        """
        get parser for a source based on url
        
        args:
            url (str): article url
            
        returns:
            function: parser function
        """
        domain = urlparse(url).netloc
        
        if 'idnes' in domain:
            return self.parse_idnes
        elif 'novinky' in domain:
            return self.parse_novinky
        elif 'irozhlas' in domain:
            return self.parse_irozhlas
        elif 'seznamzpravy' in domain:
            return self.parse_seznamzpravy
        elif 'aktualne' in domain:
            return self.parse_aktualne
        else:
            return None
    
    def scrape_articles(self, max_articles=50):
        """
        scrape articles from all sources
        
        args:
            max_articles (int): maximum number of articles to scrape
        """
        existing_urls = self.get_existing_urls()
        articles_scraped = 0
        
        for source in self.sources:
            try:
                logger.info(f"scraping articles from {source['name']}")
                articles_from_rss = self.fetch_rss_feed(source['rss_url'])
                
                # filter out existing urls
                new_articles = [a for a in articles_from_rss if a['ArticleUrl'] not in existing_urls]
                logger.info(f"found {len(new_articles)} new articles from {source['name']}")
                
                # limit number of articles per source
                if len(new_articles) > max_articles // len(self.sources):
                    new_articles = new_articles[:max_articles // len(self.sources)]
                
                for article_info in new_articles:
                    # random delay to be polite to servers
                    time.sleep(random.uniform(1, 3))
                    
                    try:
                        article_data = source['parser'](article_info['ArticleUrl'])
                        if article_data:
                            if self.save_article(article_data):
                                articles_scraped += 1
                                logger.info(f"scraped {articles_scraped} articles so far")
                                
                                if articles_scraped >= max_articles:
                                    logger.info(f"reached maximum number of articles ({max_articles})")
                                    break
                    except Exception as e:
                        logger.error(f"error processing article {article_info['ArticleUrl']}: {str(e)}")
                
                if articles_scraped >= max_articles:
                    break
                    
                # random delay between sources
                time.sleep(random.uniform(2, 5))
            except Exception as e:
                logger.error(f"error scraping source {source['name']}: {str(e)}")
        
        # save scraped articles to file
        self.save_to_file()
        
        logger.info(f"scraping completed: {articles_scraped} articles scraped")

def main():
    """main function for running the scraper"""
    # load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config', 'database.json')
    
    db_config = None
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                db_config = json.load(f)
            logger.info("loaded database configuration from config file")
        except Exception as e:
            logger.warning(f"error loading configuration: {str(e)}, using default values")
    
    # create scraper
    scraper = NewsScraper(db_config)
    
    # scrape articles
    max_articles = 50  # adjust as needed
    scraper.scrape_articles(max_articles=max_articles)
    
    # disconnect from database
    scraper.disconnect_from_db()
    
    # process the articles (preprocess and apply models)
    process_articles()

def process_articles():
    """preprocess and apply models to newly scraped articles"""
    try:
        # get path to processing script
        process_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'process_scraped_data.py')
        
        # check if script exists
        if os.path.exists(process_script):
            # execute processing script using the same python executable
            python_exe = sys.executable
            process = subprocess.Popen([python_exe, process_script], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
            
            logger.info("processing scraped articles...")
            
            # wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info("articles processing completed successfully")
            else:
                logger.error(f"articles processing failed with error: {stderr.decode('utf-8')}")
        else:
            logger.warning(f"processing script not found at: {process_script}")
    except Exception as e:
        logger.error(f"error processing articles: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"unexpected error: {str(e)}")