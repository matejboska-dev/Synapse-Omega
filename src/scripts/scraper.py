import requests
from bs4 import BeautifulSoup
import feedparser
import datetime
import time
import random
import re
import pyodbc
from tqdm import tqdm
import logging
from urllib.parse import urlparse
import concurrent.futures
import argparse
import subprocess
import os
import sys
import json

# Nastavení loggeru
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("synapse-scraper")

# Konfigurace připojení k databázi
DB_SERVER = "193.85.203.188"
DB_NAME = "boska"
DB_USER = "boska"
DB_PASSWORD = "123456"

# Seznam zdrojů zpráv a jejich RSS
# Seznam zdrojů zpráv a jejich RSS
news_sources = {
    "novinky": "https://www.novinky.cz/rss",
    "seznamzpravy": "https://www.seznamzpravy.cz/rss",
    "zpravy": "https://www.zpravy.cz/rss"
}
# Funkce pro připojení k databázi
def connect_to_db():
    """
    Vytvoří připojení k SQL Server databázi.
    Returns:
        pyodbc.Connection: Objekt připojení k databázi nebo None v případě chyby
    """
    try:
        # Try to load config from file first
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config', 'database.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                server = config.get('server', DB_SERVER)
                database = config.get('database', DB_NAME)
                username = config.get('username', DB_USER)
                password = config.get('password', DB_PASSWORD)
            logger.info("Používám konfiguraci databáze z config/database.json")
        else:
            server = DB_SERVER
            database = DB_NAME
            username = DB_USER
            password = DB_PASSWORD
            
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
        conn = pyodbc.connect(conn_str)
        logger.info("Úspěšně připojeno k databázi.")
        return conn
    except Exception as e:
        logger.error(f"Chyba při připojení k databázi: {e}")
        return None

def get_article_text(url, source, max_retries=3):
    """
    Extrahuje text článku z dané URL adresy s lepším zachováním formátování.
    
    Args:
        url (str): URL adresa článku
        source (str): Zdroj článku (např. "novinky", "seznamzpravy")
        max_retries (int): Maximální počet pokusů o stažení článku
    
    Returns:
        tuple: (text_článku, počet_znaků, počet_slov, obrázky)
    """
    for retry in range(max_retries):
        try:
            # Přidání náhodné pauzy pro omezení zátěže na server
            time.sleep(random.uniform(1, 3))
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "cs,en-US;q=0.7,en;q=0.3",
                "Referer": "https://www.google.com/"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Pokus {retry+1}/{max_retries}: Nepodařilo se stáhnout článek: {url}, status code: {response.status_code}")
                if retry == max_retries - 1:
                    return "Nepodařilo se stáhnout článek", 0, 0, []
                continue
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Odstranění nepotřebných elementů
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "form", "button"]):
                element.decompose()
            
            # Detekce zdroje podle domény
            if not source or source == "":
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                # Zjednodušená detekce zdroje na základě domény
                for src_name in news_sources:
                    if src_name.lower() in domain.lower():
                        source = src_name
                        break
            
            # Extrakce obrázků - zachytáváme URL hlavních obrázků článku
            image_urls = []
            article_images = []
            
            # Hledáme obrázky podle zdroje
            if source == "novinky":
                article_images = soup.select('div.article-detail__media-content img, div.gallery-slider__slide img, div.article-body img')
            elif source == "seznamzpravy":
                article_images = soup.select('div.e_wi img, figure img, div.article-body img')
            elif source == "zpravy":
                article_images = soup.select('div.article-detail img, div.article__image img, .article-content img')
            else:
                # Obecný výběr obrázků
                article_images = soup.select('article img, .article img, .content img, figure img')
                
            # Extrahujeme URL obrázků
            for img in article_images:
                if img.get('src') and (img.get('src').startswith('http') or img.get('src').startswith('/')):
                    # Zajistit absolutní URL
                    img_url = img.get('src')
                    if img_url.startswith('/'):
                        parsed_url = urlparse(url)
                        img_url = f"{parsed_url.scheme}://{parsed_url.netloc}{img_url}"
                    
                    # Zkontrolovat, jestli není příliš malý obrázek (ikona)
                    if img.get('width') and img.get('height'):
                        try:
                            width = int(img['width'])
                            height = int(img['height'])
                            if width < 100 or height < 100:
                                continue  # Přeskočit malé obrázky/ikony
                        except (ValueError, TypeError):
                            pass
                    
                    # Přidat URL do seznamu
                    if img_url not in image_urls:
                        image_urls.append(img_url)
            
            # Pokud máme příliš mnoho obrázků, ponecháme jen první 3
            if len(image_urls) > 3:
                image_urls = image_urls[:3]
            
            # Univerzální extrakce textu - pomocí slovníku selektorů pro jednotlivé zdroje
            selectors = {
                "novinky": {"div": ["article-body", "article-detail__body", "article-content"]},
                "seznamzpravy": {"div": ["article-body", "e_c", "article__body", "c_i", "article_content"]},
                "zpravy": {"div": ["article-detail__content", "article-body", "b-article-main", "article-content"]}
            }
            
            paragraphs = []
            
            # Extrakce textu podle zdroje
            if source in selectors:
                for element_type, class_names in selectors[source].items():
                    article_div = None
                    for class_name in class_names:
                        article_div = soup.find(element_type, class_=class_name)
                        if article_div:
                            break
                    
                    if article_div:
                        # Získat všechny odstavce
                        p_tags = article_div.find_all("p")
                        for p in p_tags:
                            # Zachovat původní formátování textu
                            text = p.get_text().strip()
                            if text:
                                paragraphs.append(text)
                        break
            
            # Pokud nenajdeme nic specifického, použijeme obecnější metodu
            if not paragraphs:
                # Pokus 1: Hledání podle typických tříd
                for class_name in ["article-body", "article-content", "news-content", "story-content", "content"]:
                    article_div = soup.find(["article", "div", "section"], class_=lambda x: x and class_name in (x.lower() if x else ""))
                    if article_div:
                        p_tags = article_div.find_all("p")
                        for p in p_tags:
                            text = p.get_text().strip()
                            if text:
                                paragraphs.append(text)
                        break
                
                # Pokus 2: Zkusit najít hlavní článek element
                if not paragraphs:
                    article_elem = soup.find("article") or soup.find("main")
                    if article_elem:
                        p_tags = article_elem.find_all("p")
                        for p in p_tags:
                            text = p.get_text().strip()
                            if text:
                                paragraphs.append(text)
            
            # Spojit odstavce s prázdným řádkem mezi nimi
            article_text = "\n\n".join(paragraphs)
            
            # Vyčistit text od duplicitních mezer
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            
            # Počet znaků a slov
            char_count = len(article_text)
            word_count = len(article_text.split())
            
            if char_count < 100:
                logger.warning(f"Málo textu extrahováno z {url}: pouze {char_count} znaků")
                if retry < max_retries - 1:
                    continue
            
            return article_text, char_count, word_count, image_urls
            
        except Exception as e:
            logger.error(f"Pokus {retry+1}/{max_retries}: Chyba při extrakci textu z {url}: {e}")
            if retry == max_retries - 1:
                return f"Chyba: {e}", 0, 0, []
            # Zvýšíme čekací dobu mezi pokusy
            time.sleep(random.uniform(3, 5))
    
    return "Nepodařilo se extrahovat text po opakovaných pokusech", 0, 0, []

def save_article_to_db(conn, source_name, title, url, pub_date, category, char_count, word_count, article_text, image_urls=None):
    """
    Uloží článek do databáze.
    
    Args:
        conn: Připojení k databázi
        source_name: Název zdroje článku
        title: Titulek článku
        url: URL článku
        pub_date: Datum publikace
        category: Kategorie článku
        char_count: Počet znaků
        word_count: Počet slov
        article_text: Text článku
        image_urls: Seznam URL obrázků článku
    
    Returns:
        bool: True při úspěchu, False při neúspěchu
    """
    try:
        cursor = conn.cursor()
        
        # Ošetření příliš dlouhých hodnot
        if len(title) > 500:
            title = title[:497] + "..."
        if len(url) > 1000:
            url = url[:997] + "..."
        if len(source_name) > 255:
            source_name = source_name[:252] + "..."
        if category and len(category) > 255:
            category = category[:252] + "..."
        
        # Aktuální datum a čas pro pole ScrapedDate
        scraped_date = datetime.datetime.now()
        
        # Check if the table exists and has the right schema
        try:
            cursor.execute("SELECT TOP 1 * FROM Articles")
            cursor.fetchall()
        except Exception as e:
            logger.warning(f"Table Articles check failed: {e}, attempting to create it")
            try:
                # Create the table if it doesn't exist
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
                    ImageUrls NVARCHAR(MAX),
                    ScrapedDate DATETIME
                )
                """)
                conn.commit()
                logger.info("Articles table created successfully")
            except Exception as create_error:
                logger.error(f"Failed to create table: {create_error}")
                # Try to save to local file as fallback
                save_to_local_json([{
                    "source": source_name,
                    "title": title,
                    "url": url,
                    "date": str(pub_date) if pub_date else None,
                    "category": category,
                    "length": char_count,
                    "words": word_count,
                    "text": article_text,
                    "images": image_urls if image_urls else [],
                    "scraped": str(scraped_date)
                }])
                return False
        
        # Připravit obrázky pro uložení (jako JSON string)
        image_urls_json = json.dumps(image_urls if image_urls else [], ensure_ascii=False)
        
        # Zkontrolovat, zda sloupec ImageUrls existuje, pokud ne, přidat ho
        try:
            cursor.execute("SELECT ImageUrls FROM Articles WHERE Id = 1")
        except:
            try:
                cursor.execute("ALTER TABLE Articles ADD ImageUrls NVARCHAR(MAX)")
                conn.commit()
                logger.info("Added ImageUrls column to Articles table")
            except Exception as e:
                logger.warning(f"Failed to add ImageUrls column: {e}")
        
        # Zkontrolovat, zda článek již existuje
        cursor.execute("SELECT COUNT(*) FROM Articles WHERE ArticleUrl = ? OR Title = ?", (url, title))
        count = cursor.fetchone()[0]
        if count > 0:
            logger.debug(f"Článek již existuje v databázi: {title}")
            return False
        
        # Vložení článku
        sql = """
        INSERT INTO Articles (SourceName, Title, ArticleUrl, PublicationDate, Category, 
                            ArticleLength, WordCount, ArticleText, ImageUrls, ScrapedDate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (source_name, title, url, pub_date, category, 
                            char_count, word_count, article_text, image_urls_json, scraped_date))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Chyba při ukládání článku do databáze: {e}")
        try:
            conn.rollback()
        except:
            pass
            
        # Save to local file if database fails
        try:
            # Ensure data directory exists
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'scraped')
            os.makedirs(data_dir, exist_ok=True)
            
            local_file = os.path.join(data_dir, f"article_{int(time.time())}.json")
            
            with open(local_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "source": source_name,
                    "title": title,
                    "url": url,
                    "date": str(pub_date) if pub_date else None,
                    "category": category,
                    "length": char_count,
                    "words": word_count,
                    "text": article_text,
                    "images": image_urls if image_urls else [],
                    "scraped": str(scraped_date)
                }, f, ensure_ascii=False)
            logger.info(f"Uloženo do lokálního souboru: {local_file}")
        except Exception as backup_e:
            logger.error(f"Také se nepodařilo uložit článek lokálně: {backup_e}")
            
        return False

# Funkce pro kontrolu existence článku v databázi (optimalizovaná o kontrolu titulku)
def article_exists(conn, url, title):
    """
    Kontroluje, zda článek již existuje v databázi podle URL nebo titulku.
    
    Args:
        conn: Připojení k databázi
        url: URL článku
        title: Titulek článku
    
    Returns:
        bool: True pokud článek existuje, jinak False
    """
    try:
        cursor = conn.cursor()
        # Kontrola podle URL nebo podobného titulku
        cursor.execute("SELECT COUNT(*) FROM Articles WHERE ArticleUrl = ? OR Title = ?", (url, title))
        count = cursor.fetchone()[0]
        return count > 0
    except Exception as e:
        logger.error(f"Chyba při kontrole existence článku: {e}")
        return False

# Funkce pro převod data z RSS na DATETIME
def parse_date(date_str):
    """
    Převede řetězec data z RSS na datetime objekt.
    
    Args:
        date_str: Řetězec s datem
    
    Returns:
        datetime: Objekt datetime nebo None
    """
    if not date_str:
        return None
    
    try:
        # RSS data mohou mít různé formáty
        for date_format in [
            "%a, %d %b %Y %H:%M:%S %z",      # Standardní RSS formát
            "%a, %d %b %Y %H:%M:%S %Z",      # Varianta s textovým timezone
            "%a, %d %b %Y %H:%M:%S GMT",     # Varianta bez timezone
            "%Y-%m-%dT%H:%M:%S%z",           # ISO formát
            "%Y-%m-%dT%H:%M:%S%Z",           # ISO s textovým timezone
            "%Y-%m-%dT%H:%M:%SZ",            # ISO bez timezone
            "%d.%m.%Y %H:%M:%S",             # Český formát
            "%d.%m.%Y",                      # Kratší český formát
        ]:
            try:
                return datetime.datetime.strptime(date_str, date_format)
            except ValueError:
                continue
                
        return datetime.datetime.now()  # Fallback pokud žádný formát nesedí
    except:
        return datetime.datetime.now()

# Funkce pro získání počtu článků v databázi
def get_article_count(conn):
    """
    Zjistí počet článků v databázi.
    
    Args:
        conn: Připojení k databázi
    
    Returns:
        int: Počet článků v databázi
    """
    try:
        if not conn:
            return 0
            
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM Articles")
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Chyba při dotazu na počet článků: {e}")
            return 0
    except Exception as e:
        logger.error(f"Chyba při získávání počtu článků: {e}")
        return 0

# Save data to local JSON file
def save_to_local_json(articles, filename=None):
    """
    Save articles to local JSON file
    
    Args:
        articles (list): List of article dictionaries
        filename (str, optional): Filename to save to. If None, generates one based on timestamp
    
    Returns:
        bool: True on success, False on failure
    """
    if not filename:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"articles_{timestamp}.json"
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'scraped')
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, filename)
    
    try:
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Uloženo {len(articles)} článků do {file_path}")
        return True
    except Exception as e:
        logger.error(f"Chyba při ukládání článků do souboru: {e}")
        return False

def process_source(source_name, rss_url, conn, current_count, target_count, max_articles_per_source, new_articles_added, daily_mode=False, newest_first=False):
    """
    Zpracuje články z jednoho zdroje.
    
    Args:
        source_name: Název zdroje
        rss_url: URL RSS feedu
        conn: Připojení k databázi
        current_count: Aktuální počet článků
        target_count: Cílový počet článků
        max_articles_per_source: Max počet článků z jednoho zdroje
        new_articles_added: Referenční proměnná pro sledování nově přidaných článků
        daily_mode: Pokud True, sbírá pouze články z aktuálního dne
        newest_first: Pokud True, seřadí články od nejnovějších
    
    Returns:
        int or list: Počet nově přidaných článků z tohoto zdroje, nebo seznam článků pokud není database
    """
    source_articles = 0
    articles_list = []
    db_connected = conn is not None
    
    # Get today's date for filtering in daily mode
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    try:
        logger.info(f"Zpracovávám RSS feed: {source_name}")
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            logger.warning(f"Žádné články nenalezeny v RSS feedu: {source_name}")
            return 0 if db_connected else articles_list
        
        # Pokud požadujeme nejnovější články, seřadíme je podle data
        entries = feed.entries
        if newest_first:
            # Získání datumů publikace
            entry_dates = []
            for entry in entries:
                pub_date = None
                if hasattr(entry, 'published'):
                    pub_date = parse_date(entry.published)
                elif hasattr(entry, 'pubDate'):
                    pub_date = parse_date(entry.pubDate)
                elif hasattr(entry, 'updated'):
                    pub_date = parse_date(entry.updated)
                
                entry_dates.append((entry, pub_date or datetime.datetime.now()))
            
            # Seřazení od nejnovějších
            entry_dates.sort(key=lambda x: x[1], reverse=True)
            entries = [entry for entry, _ in entry_dates]
        
        # Procházení článků v RSS feedu
        for entry in tqdm(entries, desc=f"Články z {source_name}"):
            # Kontrola, jestli už nemáme dostatek článků
            if current_count + new_articles_added.value >= target_count:
                logger.info(f"Dosaženo cílového počtu článků: {current_count + new_articles_added.value}")
                break
                
            # Kontrola, jestli už nemáme dostatek článků z tohoto zdroje
            if source_articles >= max_articles_per_source:
                logger.info(f"Dosaženo maximálního počtu článků ze zdroje {source_name}: {source_articles}")
                break
            
            # Získání základních údajů z RSS
            title = entry.title if hasattr(entry, 'title') else "Bez názvu"
            url = entry.link if hasattr(entry, 'link') else ""
            
            if not url:
                continue
            
            # Kontrola, zda článek už neexistuje v databázi
            if conn and article_exists(conn, url, title):
                logger.debug(f"Článek již existuje v databázi: {title}")
                continue
            
            # Zpracování datumu publikace
            pub_date = None
            if hasattr(entry, 'published'):
                pub_date = parse_date(entry.published)
            elif hasattr(entry, 'pubDate'):
                pub_date = parse_date(entry.pubDate)
            elif hasattr(entry, 'updated'):
                pub_date = parse_date(entry.updated)
            
            # If in daily mode, check if this article is from today
            if daily_mode and pub_date:
                article_date = pub_date.strftime('%Y-%m-%d')
                if article_date != today:
                    # Skip articles not from today
                    logger.debug(f"Přeskakuji starší článek (daily mode): {title} z {article_date}")
                    continue
            
            # Pokus o získání kategorie
            category = ""
            if hasattr(entry, 'tags') and entry.tags:
                try:
                    category = entry.tags[0].term
                except:
                    try:
                        category = entry.tags[0]['term']
                    except:
                        category = ""
            elif hasattr(entry, 'category'):
                category = entry.category
            
            # Získání plného textu článku
            logger.info(f"Stahuji článek: {title}")
            article_text, char_count, word_count = get_article_text(url, source_name)
            
            # Create article dictionary for either database or local storage
            article_data = {
                "source": source_name,
                "title": title,
                "url": url,
                "date": str(pub_date) if pub_date else None,
                "category": category,
                "length": char_count,
                "words": word_count,
                "text": article_text,
                "scraped": str(datetime.datetime.now())
            }
            
            # Uložení článku do databáze nebo lokálně
            if conn:
                success = save_article_to_db(
                    conn, source_name, title, url, pub_date, category, 
                    char_count, word_count, article_text
                )
            else:
                # If no database connection, collect for local saving
                articles_list.append(article_data)
                success = True
            
            if success:
                with new_articles_added.get_lock():
                    new_articles_added.value += 1
                source_articles += 1
                logger.info(f"Článek uložen: {title} ({char_count} znaků, {word_count} slov)")
            
            # Informace o postupu
            if source_articles % 5 == 0:
                logger.info(f"Celkem staženo nových článků: {new_articles_added.value}")
            
            # Krátká pauza mezi články pro snížení zátěže
            time.sleep(random.uniform(0.5, 1.5))
        
        return source_articles if db_connected else articles_list
        
    except Exception as e:
        logger.error(f"Chyba při zpracování zdroje {source_name}: {e}")
        return source_articles if db_connected else articles_list

def collect_latest_news(max_per_source=5):
    """
    Sbírá pouze nejnovější zprávy z každého zdroje - optimalizováno pro rychlost a rychlý přehled
    
    Args:
        max_per_source: Maximální počet článků z jednoho zdroje
    
    Returns:
        int: Počet nově přidaných článků
    """
    # Připojení k databázi
    conn = connect_to_db()
    db_connected = conn is not None
    
    if not db_connected:
        logger.warning("Nepodařilo se připojit k databázi. Články budou uloženy lokálně.")
        # Pokus vytvořit alespoň lokální soubor s daty
        local_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'scraped')
        os.makedirs(local_data_dir, exist_ok=True)
    
    # Inicializace sdílené proměnné pro počítání nových článků
    import multiprocessing
    new_articles_added = multiprocessing.Value('i', 0)
    all_articles = []
    
    try:
        # Use our three selected news sources
        logger.info(f"Sbírám nejnovější zprávy z vybraných zdrojů (Novinky.cz, SeznamZpravy.cz, Zpravy.cz), max {max_per_source} článků z každého zdroje")
        
        # Zpracování zdrojů
        for source_name, rss_url in news_sources.items():
            logger.info(f"Zpracovávám zdroj: {source_name}")
            articles_from_source = process_source(source_name, rss_url, conn, 0, max_per_source * len(news_sources), 
                          max_per_source, new_articles_added, daily_mode=True, newest_first=True)
            
            # If database connection failed, collect articles for local storage
            if not db_connected and isinstance(articles_from_source, list):
                all_articles.extend(articles_from_source)
    
    finally:
        if conn:
            conn.close()
            logger.info("Databáze uzavřena.")
    
    # Save all articles to a local file if database failed
    if not db_connected and all_articles:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_to_local_json(all_articles, f"articles_{timestamp}.json")
    
    logger.info(f"Sběr dokončen! Přidáno {new_articles_added.value} nových článků.")
    
    # If we couldn't connect to the database initially, try again and insert the articles
    if not db_connected and all_articles:
        conn = connect_to_db()
        if conn:
            logger.info("Podařilo se připojit k databázi na druhý pokus. Ukládám články do databáze.")
            articles_saved = 0
            for article in all_articles:
                if save_article_to_db(
                    conn,
                    article['source'],
                    article['title'],
                    article['url'],
                    article.get('date'),
                    article.get('category', ''),
                    article.get('length', 0),
                    article.get('words', 0),
                    article.get('text', '')
                ):
                    articles_saved += 1
            logger.info(f"Uloženo {articles_saved} článků do databáze na druhý pokus.")
            conn.close()
    
    return new_articles_added.value

# Hlavní funkce pro sběr článků
def collect_news(target_count=3000, max_articles_per_source=300, daily_mode=False, newest_first=False):
    """
    Sbírá články z různých zdrojů až do dosažení cílového počtu.
    
    Args:
        target_count: Cílový počet článků celkem
        max_articles_per_source: Max počet článků z jednoho zdroje
        daily_mode: Pokud True, sbírá pouze články z aktuálního dne
        newest_first: Pokud True, seřadí články od nejnovějších
    
    Returns:
        int: Počet nově přidaných článků
    """
    # Připojení k databázi
    conn = connect_to_db()
    if not conn:
        logger.error("Nelze pokračovat bez připojení k databázi.")
        return 0
    
    # Zjištění aktuálního počtu článků
    current_count = get_article_count(conn)
    logger.info(f"Aktuální počet článků v databázi: {current_count}")
    
    # Pokud již máme dostatek článků, končíme
    if current_count >= target_count:
        logger.info(f"Již máme dostatek článků ({current_count}/{target_count}), není třeba stahovat další.")
        conn.close()
        return 0
    
    # Inicializace sdílené proměnné pro počítání nových článků
    import multiprocessing
    new_articles_added = multiprocessing.Value('i', 0)
    
    try:
        # Rozdělení zdrojů pro sekvenční zpracování (paralelní by mohlo způsobit problémy s databází)
        for source_name, rss_url in news_sources.items():
            # Kontrola, jestli už nemáme dostatek článků
            if current_count + new_articles_added.value >= target_count:
                logger.info(f"Dosaženo cílového počtu článků: {current_count + new_articles_added.value}")
                break
            
            process_source(source_name, rss_url, conn, current_count, target_count, 
                           max_articles_per_source, new_articles_added, daily_mode, newest_first)
    
    finally:
        if conn:
            conn.close()
            logger.info("Připojení k databázi uzavřeno.")
    
    logger.info(f"Dokončeno! Přidáno {new_articles_added.value} nových článků.")
    logger.info(f"Aktuální celkový počet článků v databázi: {current_count + new_articles_added.value}")
    return new_articles_added.value

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='News article scraper')
    parser.add_argument('--daily', action='store_true', help='Scrape only today\'s headlines')
    parser.add_argument('--latest', action='store_true', help='Scrape only latest headlines')
    parser.add_argument('--max-articles', type=int, default=3000, help='Maximum number of articles to scrape')
    parser.add_argument('--max-per-source', type=int, help='Maximum number of articles per source')
    parser.add_argument('--database', action='store_true', help='Force database storage (will retry if fails)')
    args = parser.parse_args()
    
    logger.info("Začínám sběr zpravodajských článků pro projekt Synapse...")
    logger.info(f"Používané zdroje: {', '.join(news_sources.keys())}")
    
    # Check database connection if database flag is set
    if args.database:
        db_conn = connect_to_db()
        if db_conn:
            logger.info("Úspěšné připojení k databázi, články budou ukládány do DB")
            db_conn.close()
        else:
            logger.warning("Nepodařilo se připojit k databázi, zkusím to znovu později")
    
    # For latest mode, we just want a few articles from each source
    if args.latest:
        logger.info("Spuštěn režim nejnovějších zpráv")
        max_per_source = args.max_per_source if args.max_per_source else 10
        # Use optimized latest news collection
        collected = collect_latest_news(max_per_source=max_per_source)
        logger.info(f"Sběr dokončen, přidáno {collected} nových článků")
    elif args.daily:
        # Daily mode - articles from today
        logger.info("Spuštěn denní režim - pouze dnešní hlavní zprávy")
        target_count = args.max_articles
        max_per_source = args.max_per_source if args.max_per_source else 10
        collected = collect_news(target_count=target_count, max_articles_per_source=max_per_source, 
                               daily_mode=True, newest_first=True)
        logger.info(f"Sběr dokončen, přidáno {collected} nových článků")
    else:
        # Regular mode - comprehensive scraping
        target_count = args.max_articles
        max_per_source = args.max_per_source if args.max_per_source else 300
        
        # Check current article count first
        current_count = get_article_count(connect_to_db())
        if current_count >= target_count:
            logger.info(f"Cílový počet článků již dosažen: {current_count}/{target_count}")
        else:
            logger.info(f"Pokračuji ve sběru, aktuálně máme {current_count}/{target_count} článků")
            collected = collect_news(target_count=target_count, max_articles_per_source=max_per_source)
            logger.info(f"Sběr dokončen, celkový počet článků: {get_article_count(connect_to_db())}")
    
    # Process scraped articles
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
            
            logger.info("Zpracování nasbíraných článků...")
            
            # wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=60)  # 60 seconds timeout
                
                if process.returncode == 0:
                    logger.info("Zpracování článků úspěšně dokončeno")
                else:
                    logger.error(f"Zpracování článků selhalo s chybou: {stderr.decode('utf-8')}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("Zpracování článků přerušeno po 60 sekundách")
                
        else:
            logger.warning(f"Skript pro zpracování nebyl nalezen na cestě: {process_script}")
    except Exception as e:
        logger.error(f"Chyba při zpracování článků: {str(e)}")