import pyodbc
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Třída pro připojení k SQL Server databázi a načtení dat článků.
    """
    
    def __init__(self, server, database, username, password, driver='{ODBC Driver 17 for SQL Server}'):
        """
        Inicializace připojení k databázi.
        
        Args:
            server (str): Adresa SQL serveru
            database (str): Název databáze
            username (str): Uživatelské jméno
            password (str): Heslo
            driver (str): ODBC driver
        """
        self.connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        self.conn = None
        
    def connect(self):
        """Připojení k databázi"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            logger.info("Úspěšně připojeno k databázi")
            return True
        except Exception as e:
            logger.error(f"Chyba při připojování k databázi: {str(e)}")
            return False
            
    def disconnect(self):
        """Odpojení od databáze"""
        if self.conn:
            self.conn.close()
            logger.info("Odpojeno od databáze")
            
    def load_articles(self, limit=None):
        """
        Načtení článků z databáze.
        
        Args:
            limit (int, optional): Maximální počet článků k načtení
        
        Returns:
            pandas.DataFrame: DataFrame s daty článků
        """
        if not self.conn:
            if not self.connect():
                return None
                
        try:
            # Zjistíme strukturu tabulky Articles
            cursor = self.conn.cursor()
            cursor.execute("SELECT TOP 1 * FROM Articles")
            columns = [column[0] for column in cursor.description]
            
            # Sestavíme dotaz podle dostupných sloupců
            query = f"SELECT * FROM Articles"
            if limit:
                query += f" TOP {limit}"
                
            # Provedeme dotaz a načteme data
            logger.info(f"Načítám články z databáze{' (limit: ' + str(limit) + ')' if limit else ''}")
            df = pd.read_sql(query, self.conn)
            
            logger.info(f"Načteno {len(df)} článků")
            return df
        except Exception as e:
            logger.error(f"Chyba při načítání článků: {str(e)}")
            return None