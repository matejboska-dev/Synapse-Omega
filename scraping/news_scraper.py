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
news_sources = {
    "idnes": "https://servis.idnes.cz/rss.aspx?c=zpravodaj",
    "novinky": "https://www.novinky.cz/rss",
    "seznamzpravy": "https://www.seznamzpravy.cz/rss",
    "aktualne": "https://zpravy.aktualne.cz/rss/",
    "ihned": "https://ihned.cz/rss/",
    "denik-n": "https://denikn.cz/feed/",
    "ct24": "https://ct24.ceskatelevize.cz/rss/hlavni-zpravy",
    "irozhlas": "https://www.irozhlas.cz/rss/irozhlas/",
    "denik": "https://www.denik.cz/rss/all.html",
    "lidovky": "https://servis.lidovky.cz/rss.aspx?c=ln_domov",
    "reflex": "https://www.reflex.cz/rss",
    "echo24": "https://echo24.cz/rss",
    # Přidáno více zdrojů pro získání většího množství dat
    "info-cz": "https://www.info.cz/feed/rss",
    "forum24": "https://www.forum24.cz/feed/",
    "blesk": "https://www.blesk.cz/rss",
    "cnn-iprima": "https://cnn.iprima.cz/rss",
    "parlamentnilisty": "https://www.parlamentnilisty.cz/export/rss.aspx",
    "eurozpravy": "https://eurozpravy.cz/rss/",
    "tyden": "https://www.tyden.cz/rss/",
    "e15": "https://www.e15.cz/rss"
}

# Funkce pro připojení k databázi
def connect_to_db():
    """
    Vytvoří připojení k SQL Server databázi.
    Returns:
        pyodbc.Connection: Objekt připojení k databázi nebo None v případě chyby
    """
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD}"
    try:
        conn = pyodbc.connect(conn_str)
        logger.info("Úspěšně připojeno k databázi.")
        return conn
    except Exception as e:
        logger.error(f"Chyba při připojení k databázi: {e}")
        return None

# Funkce pro získání textu článku s retry logikou
def get_article_text(url, source, max_retries=3):
    """
    Extrahuje text článku z dané URL adresy.
    
    Args:
        url (str): URL adresa článku
        source (str): Zdroj článku (např. "idnes", "novinky")
        max_retries (int): Maximální počet pokusů o stažení článku
    
    Returns:
        tuple: (text_článku, počet_znaků, počet_slov)
    """
    for retry in range(max_retries):
        try:
            # Přidání náhodné pauzy pro omezení zátěže na server
            time.sleep(random.uniform(1, 3))
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "cs,en-US;q=0.7,en;q=0.3",
                "Referer": "https://www.google.com/"  # Přidáno pro větší autenticitu
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Pokus {retry+1}/{max_retries}: Nepodařilo se stáhnout článek: {url}, status code: {response.status_code}")
                if retry == max_retries - 1:
                    return "Nepodařilo se stáhnout článek", 0, 0
                continue
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Odstranění nepotřebných elementů
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "form", "button"]):
                element.decompose()
            
            # Detekce zdroje podle domény, pokud není explicitně uvedeno
            if not source or source == "":
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                # Zjednodušená detekce zdroje na základě domény
                for src_name in news_sources:
                    if src_name.lower() in domain.lower():
                        source = src_name
                        break
            
            # Univerzální extrakce textu - pomocí slovníku selektorů pro jednotlivé zdroje
            selectors = {
                "idnes": {"div": ["article-body"]},
                "novinky": {"div": ["articleBody", "article_text"]},
                "seznamzpravy": {"div": ["article-body", "sznp-article-body"]},
                "aktualne": {"div": ["article-text", "clanek"]},
                "ihned": {"div": ["article-body", "clanek"]},
                "denik-n": {"div": ["post-content", "a_content"]},
                "ct24": {"div": ["article-body", "article_text"]},
                "irozhlas": {"div": ["b-detail", "article"]},
                "denik": {"div": ["article-body", "clanek"]},
                "lidovky": {"div": ["article-body"]},
                "reflex": {"div": ["article-content"]},
                "echo24": {"div": ["article-detail__content", "article_body"]},
                "info-cz": {"div": ["article-body"]},
                "forum24": {"div": ["entry-content"]},
                "blesk": {"div": ["o-article__content"]},
                "cnn-iprima": {"div": ["article-content"]},
                "parlamentnilisty": {"div": ["article-body"]},
                "eurozpravy": {"div": ["article-content"]},
                "tyden": {"div": ["text-block"]},
                "e15": {"div": ["article-body"]}
            }
            
            # Extrakce textu podle zdroje
            article_text = ""
            if source in selectors:
                for element_type, class_names in selectors[source].items():
                    for class_name in class_names:
                        if article_div := soup.find(element_type, class_=class_name):
                            paragraphs = article_div.find_all("p")
                            article_text = " ".join([p.get_text().strip() for p in paragraphs])
                            break
                    if article_text:
                        break
            
            # Obecná metoda jako fallback
            if not article_text:
                # Pokus 1: Hledání podle typických tříd
                for class_name in ["article-body", "article-content", "post-content", "news-content", "story-content", "main-content", "entry-content", "article", "clanek"]:
                    if article_div := soup.find(["div", "article", "section"], class_=lambda x: x and class_name in x.lower()):
                        paragraphs = article_div.find_all("p")
                        article_text = " ".join([p.get_text().strip() for p in paragraphs])
                        break
                
                # Pokus 2: Hledání podle typických HTML elementů
                if not article_text:
                    main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=lambda x: x and any(c in (x.lower() if x else "") for c in ["content", "article", "text", "body"]))
                    if main_content:
                        paragraphs = main_content.find_all("p")
                        article_text = " ".join([p.get_text().strip() for p in paragraphs])
                    else:
                        # Poslední pokus - vše z body kromě vyloučených elementů
                        body = soup.find("body")
                        if body:
                            article_text = body.get_text(separator=" ", strip=True)
            
            # Čištění textu
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            article_text = re.sub(r'[^\w\s.,?!;:()\[\]{}"\'–—-]', '', article_text)
            
            # Počet znaků a slov
            char_count = len(article_text)
            word_count = len(article_text.split())
            
            if char_count < 100:
                logger.warning(f"Málo textu extrahováno z {url}: pouze {char_count} znaků")
                if retry < max_retries - 1:
                    continue
            
            return article_text, char_count, word_count
            
        except Exception as e:
            logger.error(f"Pokus {retry+1}/{max_retries}: Chyba při extrakci textu z {url}: {e}")
            if retry == max_retries - 1:
                return f"Chyba: {e}", 0, 0
            # Zvýšíme čekací dobu mezi pokusy pro snížení pravděpodobnosti blokování
            time.sleep(random.uniform(3, 5))
    
    return "Nepodařilo se extrahovat text po opakovaných pokusech", 0, 0

# Funkce pro uložení článku do databáze
def save_article_to_db(conn, source_name, title, url, pub_date, category, char_count, word_count, article_text):
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
        
        sql = """
        INSERT INTO Articles (SourceName, Title, ArticleUrl, PublicationDate, Category, 
                              ArticleLength, WordCount, ArticleText, ScrapedDate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (source_name, title, url, pub_date, category, 
                             char_count, word_count, article_text, scraped_date))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Chyba při ukládání článku do databáze: {e}")
        try:
            conn.rollback()
        except:
            pass
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
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Articles")
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        logger.error(f"Chyba při získávání počtu článků: {e}")
        return 0

# Funkce pro zpracování článků z jednoho zdroje
def process_source(source_name, rss_url, conn, current_count, target_count, max_articles_per_source, new_articles_added):
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
    
    Returns:
        int: Počet nově přidaných článků z tohoto zdroje
    """
    source_articles = 0
    
    try:
        logger.info(f"Zpracovávám RSS feed: {source_name}")
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            logger.warning(f"Žádné články nenalezeny v RSS feedu: {source_name}")
            return 0
        
        # Procházení článků v RSS feedu
        for entry in tqdm(feed.entries, desc=f"Články z {source_name}"):
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
            if article_exists(conn, url, title):
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
            
            # Kontrola, zda se podařilo získat dostatek textu
            if char_count < 100:
                logger.warning(f"Příliš málo textu ze článku: {title}, přeskakuji")
                continue
            
            # Uložení článku do databáze
            success = save_article_to_db(
                conn, source_name, title, url, pub_date, category, 
                char_count, word_count, article_text
            )
            
            if success:
                with new_articles_added.get_lock():
                    new_articles_added.value += 1
                source_articles += 1
                logger.info(f"Článek uložen: {title} ({char_count} znaků, {word_count} slov)")
            
            # Informace o postupu
            if source_articles % 10 == 0:
                logger.info(f"Celkem staženo nových článků: {new_articles_added.value}")
            
            # Krátká pauza mezi články pro snížení zátěže
            time.sleep(random.uniform(0.5, 1.5))
        
        return source_articles
        
    except Exception as e:
        logger.error(f"Chyba při zpracování zdroje {source_name}: {e}")
        return source_articles

# Hlavní funkce pro sběr článků
def collect_news(target_count=3000, max_articles_per_source=300):
    """
    Sbírá články z různých zdrojů až do dosažení cílového počtu.
    
    Args:
        target_count: Cílový počet článků celkem
        max_articles_per_source: Max počet článků z jednoho zdroje
    
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
                           max_articles_per_source, new_articles_added)
    
    finally:
        if conn:
            conn.close()
            logger.info("Připojení k databázi uzavřeno.")
    
    logger.info(f"Dokončeno! Přidáno {new_articles_added.value} nových článků.")
    logger.info(f"Aktuální celkový počet článků v databázi: {current_count + new_articles_added.value}")
    return new_articles_added.value

# Hlavní spuštění programu
if __name__ == "__main__":
    logger.info("Začínám sběr zpravodajských článků pro projekt Synapse...")
    
    # Cílový počet článků (změněno na 3000)
    target_count = 3000
    
    # Maximální počet článků z jednoho zdroje (pro rovnoměrnější distribuci)
    max_per_source = 300
    
    # Opakované spouštění dokud nedosáhneme cílového počtu
    while True:
        current_count = get_article_count(connect_to_db())
        if current_count >= target_count:
            logger.info(f"Cílový počet článků dosažen: {current_count}/{target_count}")
            break
            
        logger.info(f"Pokračuji ve sběru, aktuálně máme {current_count}/{target_count} článků")
        collected = collect_news(target_count=target_count, max_articles_per_source=max_per_source)
        
        if collected == 0:
            logger.warning("Žádné nové články nebyly přidány, možná jsme již vyčerpali dostupné zdroje.")
            # Pokud nepřibyly žádné články, počkáme 30 minut před dalším pokusem
            logger.info("Čekám 30 minut před dalším pokusem...")
            time.sleep(1800)  # 30 minut
    
    logger.info(f"Sběr dokončen, cílový počet článků dosažen: {get_article_count(connect_to_db())}")