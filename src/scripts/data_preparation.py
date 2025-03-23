import sys
import os
import logging
import pandas as pd
from datetime import datetime

# Přidání nadřazené složky do systémové cesty pro import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database_connector import DatabaseConnector
from data.data_analyzer import DataAnalyzer
from data.text_preprocessor import TextPreprocessor

# Konfigurace loggeru
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Hlavní funkce pro analýzu a předzpracování dat.
    """
    # Vytvoření adresářů pro výstup, pokud neexistují
    for directory in ['logs', 'data/processed', 'reports/figures']:
        os.makedirs(directory, exist_ok=True)
    
    # Připojení k databázi a načtení článků
    logger.info("== Připojení k databázi a načtení dat ==")
    
    # Parametry připojení
    server = "localhost"
    database = "NewsDB"
    username = "user"
    password = "password"
    
    # TODO: Načtení parametrů připojení z konfiguračního souboru nebo prostředí
    try:
        # Pokus o načtení konfigurace
        import json
        with open('config/database.json', 'r') as f:
            config = json.load(f)
            server = config.get('server', server)
            database = config.get('database', database)
            username = config.get('username', username)
            password = config.get('password', password)
        logger.info("Načtena konfigurace z config/database.json")
    except:
        logger.warning("Konfigurace nebyla načtena, používám výchozí hodnoty")
    
    # Vytvoření instance konektoru a připojení k databázi
    db = DatabaseConnector(server, database, username, password)
    if not db.connect():
        logger.error("Nelze se připojit k databázi, končím")
        return
    
    # Načtení článků
    df = db.load_articles()
    if df is None or df.empty:
        logger.error("Nepodařilo se načíst data, končím")
        db.disconnect()
        return
    
    # Základní informace o datech
    logger.info(f"Načteno {len(df)} článků s {len(df.columns)} sloupci")
    logger.info(f"Sloupce: {', '.join(df.columns)}")
    
    # Uložení surových dat
    raw_data_path = 'data/raw/articles_raw.csv'
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    df.to_csv(raw_data_path, index=False, encoding='utf-8')
    logger.info(f"Surová data uložena do {raw_data_path}")
    
    # Analýza dat
    logger.info("== Analýza dat ==")
    analyzer = DataAnalyzer(df)
    stats = analyzer.compute_basic_stats()
    
    # Výpis základních statistik
    logger.info(f"Celkový počet článků: {stats['total_articles']}")
    
    if 'articles_by_source' in stats:
        logger.info(f"Počet zdrojů: {stats['num_sources']}")
        for source, count in sorted(stats['articles_by_source'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {source}: {count} článků")
    
    if 'articles_by_category' in stats:
        logger.info(f"Počet kategorií: {stats['num_categories']}")
        for category, count in sorted(stats['articles_by_category'].items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {category}: {count} článků")
    
    if 'date_range' in stats:
        logger.info(f"Časové rozpětí: {stats['date_range']['min']} až {stats['date_range']['max']}")
    
    if 'content_length' in stats:
        logger.info(f"Průměrná délka článku: {stats['content_length']['mean']:.1f} znaků")
        logger.info(f"Medián délky článku: {stats['content_length']['median']:.1f} znaků")
    
    if 'word_count' in stats:
        logger.info(f"Průměrný počet slov: {stats['word_count']['mean']:.1f}")
        logger.info(f"Medián počtu slov: {stats['word_count']['median']:.1f}")
    
    # Uložení statistik
    import json
    with open('reports/data_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Statistiky uloženy do reports/data_stats.json")
    
    # Vizualizace základních statistik
    analyzer.visualize_basic_stats()
    
    # Předzpracování textů
    logger.info("== Předzpracování textů ==")
    preprocessor = TextPreprocessor(language='czech')
    
    # Předzpracování titulků a obsahů
    if 'Title' in df.columns:
        df = preprocessor.preprocess_dataframe(df, 'Title')
    
    if 'Content' in df.columns:
        df = preprocessor.preprocess_dataframe(df, 'Content')
    
    # Uložení předzpracovaných dat
    processed_data_path = 'data/processed/articles_processed.csv'
    df.to_csv(processed_data_path, index=False, encoding='utf-8')
    logger.info(f"Předzpracovaná data uložena do {processed_data_path}")
    
    # Odpojení od databáze
    db.disconnect()
    
    logger.info("Analýza a předzpracování dat dokončeno")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Neočekávaná chyba: %s", str(e))