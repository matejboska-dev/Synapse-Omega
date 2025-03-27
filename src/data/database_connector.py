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
            return True
        except Exception as e:
            logger.error(f"error connecting to database: {str(e)}")
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
            
            # build query based on available columns
            query = "SELECT * FROM Articles"
            if limit:
                query = f"SELECT TOP {limit} * FROM Articles"
                
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