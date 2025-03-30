# src/scripts/download_nltk_resources.py
import nltk

def download_all_resources():
    print("Stahování NLTK zdrojů...")
    
    # Základní tokenovací nástroje
    nltk.download('punkt')
    
    # Stopslova pro češtinu a angličtinu
    nltk.download('stopwords')
    
    # Další potřebné zdroje
    nltk.download('punkt_tab')  # pro anglický tokenizer
    nltk.download('snowball_data')  # pro stemming
    
    print("Stahování dokončeno.")

if __name__ == "__main__":
    download_all_resources()