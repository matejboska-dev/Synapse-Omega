import pyodbc
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    class for connecting to SQL Server database and loading article data
    """
    
    def __init__(self, server, database, username, password, driver='{ODBC Driver 17 for SQL Server}'):
        """
        initialize database connection
        
        args:
            server (str): SQL server address
            database (str): database name
            username (str): username
            password (str): password
            driver (str): ODBC driver
        """
        self.connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        self.conn = None
        
    def connect(self):
        """connect to database"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            logger.info("successfully connected to database")
            
            # Ensure all necessary tables exist
            self.ensure_tables_exist()
            
            return True
        except Exception as e:
            logger.error(f"error connecting to database: {str(e)}")
            return False
    
    def ensure_tables_exist(self):
        """
        Ensure that the necessary tables exist in the database.
        Create them if they don't.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # Check if Articles table exists
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Articles')
            CREATE TABLE Articles (
                Id INT IDENTITY(1,1) PRIMARY KEY,
                SourceName NVARCHAR(255),
                Title NVARCHAR(500),
                ArticleUrl NVARCHAR(1000),
                PublicationDate DATETIME,
                Category NVARCHAR(255),
                ArticleLength INT,
                WordCount INT,
                ArticleText NVARCHAR(MAX),
                ScrapedDate DATETIME
            )
            """)
            self.conn.commit()
            
            logger.info("Database tables verified/created")
            return True
        except Exception as e:
            logger.error(f"Error ensuring tables exist: {str(e)}")
            return False
            
    def disconnect(self):
        """disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("disconnected from database")
            
    def load_articles(self, limit=None):
        """
        load articles from database
        
        args:
            limit (int, optional): maximum number of articles to load
        
        returns:
            pandas.DataFrame: dataframe with article data
        """
        if not self.conn:
            if not self.connect():
                return None
                
        try:
            # determine available tables first
            cursor = self.conn.cursor()
            tables = []
            for row in cursor.tables():
                if row.table_type == 'TABLE':
                    tables.append(row.table_name)
            
            # check if Articles table exists
            if 'Articles' not in tables:
                logger.error(f"Articles table not found in database. Available tables: {', '.join(tables)}")
                return None
                
            # determine the structure of the Articles table
            cursor.execute("SELECT TOP 1 * FROM Articles")
            columns = [column[0] for column in cursor.description]
            cursor.fetchall()  # clear results
            
            # build query based on available columns with ordering by publication date
            if limit:
                query = f"SELECT TOP {limit} * FROM Articles ORDER BY PublicationDate DESC"
            else:
                query = "SELECT * FROM Articles ORDER BY PublicationDate DESC"
                
            # execute query and load data
            logger.info(f"loading articles from database{' (limit: ' + str(limit) + ')' if limit else ''}")
            
            # using direct cursor to avoid pandas ODBC warning
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # convert to dataframe
            df = pd.DataFrame.from_records(rows, columns=columns)
            
            logger.info(f"loaded {len(df)} articles")
            return df
        except Exception as e:
            logger.error(f"error loading articles: {str(e)}")
            return None

    def save_article(self, source_name, title, url, pub_date, category, char_count, word_count, article_text):
        """
        Save an article to the database
        
        Args:
            source_name (str): Name of the news source
            title (str): Article title
            url (str): URL of the article
            pub_date (datetime): Publication date
            category (str): Article category
            char_count (int): Article length in characters
            word_count (int): Article word count
            article_text (str): Full text of the article
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.conn:
            if not self.connect():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # Ensure table exists
            self.ensure_tables_exist()
            
            # Check if article already exists
            cursor.execute("SELECT COUNT(*) FROM Articles WHERE ArticleUrl = ? OR Title = ?", (url, title))
            count = cursor.fetchone()[0]
            if count > 0:
                logger.debug(f"Article already exists in database: {title}")
                return False
            
            # Current timestamp for scraped date
            scraped_date = pd.Timestamp.now()
            
            # Truncate long values
            if len(title) > 500:
                title = title[:497] + "..."
            if len(url) > 1000:
                url = url[:997] + "..."
            if len(source_name) > 255:
                source_name = source_name[:252] + "..."
            if category and len(category) > 255:
                category = category[:252] + "..."
            
            # Insert article
            sql = """
            INSERT INTO Articles (SourceName, Title, ArticleUrl, PublicationDate, Category, 
                              ArticleLength, WordCount, ArticleText, ScrapedDate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(sql, (source_name, title, url, pub_date, category, 
                             char_count, word_count, article_text, scraped_date))
            self.conn.commit()
            
            logger.info(f"Article saved to database: {title}")
            return True
        except Exception as e:
            logger.error(f"Error saving article to database: {str(e)}")
            try:
                self.conn.rollback()
            except:
                pass
            return False

    def execute_query(self, query, params=None):
        """
        Execute a custom SQL query
        
        Args:
            query (str): SQL query to execute
            params (tuple, optional): Parameters for the query
            
        Returns:
            list: Query results or None on error
        """
        if not self.conn:
            if not self.connect():
                return None
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # Check if this is a SELECT query
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            else:
                self.conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            try:
                self.conn.rollback()
            except:
                pass
            return None