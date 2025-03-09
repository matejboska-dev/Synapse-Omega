import requests
from bs4 import BeautifulSoup
import feedparser
import datetime
import time
import random
import re
import pyodbc
from tqdm import tqdm
import os
import logging
from urllib.parse import urlparse

# pip install requests beautifulsoup4 feedparser pyodbc tqdm

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
}

# Funkce pro připojení k databázi
def connect_to_db():
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD}"
    try:
        conn = pyodbc.connect(conn_str)
        logger.info("Úspěšně připojeno k databázi.")
        return conn
    except Exception as e:
        logger.error(f"Chyba při připojení k databázi: {e}")
        return None

# Funkce pro získání textu článku
def get_article_text(url, source):
    try:
        # Přidání náhodné pauzy, aby server nebyl přetížen
        time.sleep(random.uniform(1, 3))
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "cs,en-US;q=0.7,en;q=0.3",
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.warning(f"Nepodařilo se stáhnout článek: {url}, status code: {response.status_code}")
            return "Nepodařilo se stáhnout článek", 0, 0
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Odstranění nepotřebných elementů
        for script in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            script.decompose()
        
        # Přizpůsobeno různým zdrojům
        article_text = ""
        
        # Detekce zdroje podle domény, pokud není explicitně uvedeno
        if not source or source == "":
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if "idnes" in domain:
                source = "idnes"
            elif "novinky" in domain:
                source = "novinky"
            elif "seznamzpravy" in domain:
                source = "seznamzpravy"
            elif "aktualne" in domain:
                source = "aktualne"
            elif "ihned" in domain:
                source = "ihned"
            elif "denikn" in domain:
                source = "denik-n"
            elif "ct24" in domain:
                source = "ct24"
            elif "irozhlas" in domain:
                source = "irozhlas"
            elif "denik.cz" in domain:
                source = "denik"
            elif "lidovky" in domain:
                source = "lidovky"
            elif "reflex" in domain:
                source = "reflex"
            elif "echo24" in domain:
                source = "echo24"
        
        # iDNES
        if source == "idnes":
            article_div = soup.find("div", class_="article-body")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Novinky
        elif source == "novinky":
            article_div = soup.find("div", class_="articleBody")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Seznam Zprávy
        elif source == "seznamzpravy":
            article_div = soup.find("div", class_="article-body")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Aktuálně
        elif source == "aktualne":
            article_div = soup.find("div", class_="article-text")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # iHNed
        elif source == "ihned":
            article_div = soup.find("div", class_="article-body")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Deník N
        elif source == "denik-n":
            article_div = soup.find("div", class_="post-content")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # ČT24
        elif source == "ct24":
            article_div = soup.find("div", class_="article-body")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # iRozhlas
        elif source == "irozhlas":
            article_div = soup.find("div", class_="b-detail")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Deník
        elif source == "denik":
            article_div = soup.find("div", class_="article-body")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Lidovky
        elif source == "lidovky":
            article_div = soup.find("div", class_="article-body")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Reflex
        elif source == "reflex":
            article_div = soup.find("div", class_="article-content")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Echo24
        elif source == "echo24":
            article_div = soup.find("div", class_="article-detail__content")
            if article_div:
                paragraphs = article_div.find_all("p")
                article_text = " ".join([p.get_text().strip() for p in paragraphs])
        
        # Obecná metoda jako fallback
        if not article_text:
            # Pokus 1: Hledání podle typických tříd
            for class_name in ["article-body", "article-content", "post-content", "news-content", "story-content", "main-content"]:
                article_div = soup.find("div", class_=class_name)
                if article_div:
                    paragraphs = article_div.find_all("p")
                    article_text = " ".join([p.get_text().strip() for p in paragraphs])
                    break
            
            # Pokus 2: Hledání podle typických HTML elementů
            if not article_text:
                main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=["content", "article", "main"])
                if main_content:
                    paragraphs = main_content.find_all("p")
                    article_text = " ".join([p.get_text().strip() for p in paragraphs])
                else:
                    # Poslední pokus - vše z body kromě skriptů a stylů
                    article_text = soup.body.get_text(separator=" ", strip=True)
        
        # Čištění textu
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        article_text = re.sub(r'[^\w\s.,?!;:()\[\]{}"\'–—-]', '', article_text)  # Odstranění speciálních znaků
        
        # Počet znaků a slov
        char_count = len(article_text)
        word_count = len(article_text.split())
        
        if char_count < 100:  # Pravděpodobně se nepodařilo správně extrahovat text
            logger.warning(f"Málo textu extrahováno z {url}: pouze {char_count} znaků")
        
        return article_text, char_count, word_count
        
    except Exception as e:
        logger.error(f"Chyba při extrakci textu z {url}: {e}")
        return f"Chyba: {e}", 0, 0

# Funkce pro uložení článku do databáze
def save_article_to_db(conn, source_name, title, url, pub_date, category, char_count, word_count, article_text):
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
        
        sql = """
        INSERT INTO Articles (SourceName, Title, ArticleUrl, PublicationDate, Category, 
                              ArticleLength, WordCount, ArticleText)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (source_name, title, url, pub_date, category, char_count, word_count, article_text))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Chyba při ukládání článku do databáze: {e}")
        conn.rollback()
        return False

# Funkce pro kontrolu, zda článek již existuje v databázi
def article_exists(conn, url):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Articles WHERE ArticleUrl = ?", (url,))
        count = cursor.fetchone()[0]
        return count > 0
    except Exception as e:
        logger.error(f"Chyba při kontrole existence článku: {e}")
        return False

# Funkce pro převod data z RSS na DATETIME
def parse_date(date_str):
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
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Articles")
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        logger.error(f"Chyba při získávání počtu článků: {e}")
        return 0

# Hlavní funkce pro sběr článků
def collect_news(target_count=1500, max_articles_per_source=300):
    # Připojení k databázi
    conn = connect_to_db()
    if not conn:
        logger.error("Nelze pokračovat bez připojení k databázi.")
        return
    
    # Zjištění aktuálního počtu článků
    current_count = get_article_count(conn)
    logger.info(f"Aktuální počet článků v databázi: {current_count}")
    
    total_articles = 0
    new_articles = 0
    
    try:
        # Procházení RSS zdrojů
        for source_name, rss_url in tqdm(news_sources.items(), desc="Zpracování zdrojů"):
            # Kontrola, jestli už nemáme dostatek článků
            if current_count + new_articles >= target_count:
                logger.info(f"Dosaženo cílového počtu článků: {current_count + new_articles}")
                break
                
            # Kontrola, jestli už nemáme dostatek článků z tohoto zdroje
            articles_from_source = 0
            
            try:
                logger.info(f"Zpracovávám RSS feed: {source_name}")
                feed = feedparser.parse(rss_url)
                
                # Procházení článků v RSS feedu
                for entry in tqdm(feed.entries, desc=f"Články z {source_name}"):
                    total_articles += 1
                    
                    # Kontrola, jestli už nemáme dostatek článků
                    if current_count + new_articles >= target_count:
                        logger.info(f"Dosaženo cílového počtu článků: {current_count + new_articles}")
                        break
                        
                    # Kontrola, jestli už nemáme dostatek článků z tohoto zdroje
                    if articles_from_source >= max_articles_per_source:
                        logger.info(f"Dosaženo maximálního počtu článků ze zdroje {source_name}: {articles_from_source}")
                        break
                    
                    # Získání základních údajů z RSS
                    title = entry.title
                    url = entry.link
                    
                    # Kontrola, zda článek už neexistuje v databázi
                    if article_exists(conn, url):
                        logger.debug(f"Článek již existuje v databázi: {title}")
                        continue
                    
                    # Zpracování datumu publikace
                    pub_date = None
                    if 'published' in entry:
                        pub_date = parse_date(entry.published)
                    elif 'pubDate' in entry:
                        pub_date = parse_date(entry.pubDate)
                    elif 'updated' in entry:
                        pub_date = parse_date(entry.updated)
                    
                    # Pokus o získání kategorie
                    category = ""
                    if 'tags' in entry and entry.tags:
                        try:
                            category = entry.tags[0].term
                        except:
                            try:
                                category = entry.tags[0]['term']
                            except:
                                category = ""
                    elif 'category' in entry:
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
                        new_articles += 1
                        articles_from_source += 1
                        logger.info(f"Článek uložen: {title} ({char_count} znaků, {word_count} slov)")
                    
                    # Informace o postupu
                    logger.info(f"Celkem staženo nových článků: {new_articles}")
                    
            except Exception as e:
                logger.error(f"Chyba při zpracování zdroje {source_name}: {e}")
    
    finally:
        if conn:
            conn.close()
            logger.info("Připojení k databázi uzavřeno.")
    
    logger.info(f"Dokončeno! Zpracováno celkem {total_articles} článků, přidáno {new_articles} nových.")
    logger.info(f"Aktuální celkový počet článků v databázi: {current_count + new_articles}")
    return new_articles

# Hlavní spuštění programu
if __name__ == "__main__":
    logger.info("Začínám sběr zpravodajských článků pro projekt Synapse...")
    
    # Cílový počet článků (zadejte požadovaný počet)
    target_count = 1500
    
    # Maximální počet článků z jednoho zdroje (pro rovnoměrnější distribuci)
    max_per_source = 300
    
    collected = collect_news(target_count=target_count, max_articles_per_source=max_per_source)
    
    logger.info(f"Sběr dokončen, přidáno {collected} nových článků.")