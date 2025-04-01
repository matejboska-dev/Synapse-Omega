import sys
import os

# Přidání správných cest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Jděte o dvě úrovně výš

# Přidejte cestu ke src
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Sys path: {sys.path}")

try:
    # Nejprve zkuste importovat z src.models
    from src.models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
    print("Import úspěšný přes src.models")
except ImportError:
    try:
        # Pak zkuste přímo z models
        from models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
        print("Import úspěšný přes models")
    except ImportError:
        print("Nelze importovat EnhancedSentimentAnalyzer, hledám dostupné moduly...")
        # Zkuste najít modul v adresáři
        models_dir = os.path.join(src_path, 'models')
        if os.path.exists(models_dir):
            print(f"Adresář models existuje: {models_dir}")
            print("Dostupné soubory:")
            for file in os.listdir(models_dir):
                print(f" - {file}")
        else:
            print(f"Adresář models neexistuje na cestě: {models_dir}")
        
        raise

# Cesta k modelu - upraveno na relativní cestu
model_path = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')
print(f"Hledám model na cestě: {model_path}")

# Zkontrolujte, zda adresář a soubory existují
if os.path.exists(model_path):
    print(f"Adresář modelu existuje: {model_path}")
    print("Soubory v adresáři modelu:")
    for file in os.listdir(model_path):
        print(f" - {file}")
else:
    print(f"Adresář modelu neexistuje: {model_path}")

# Pokus o načtení modelu
try:
    sentiment_model = EnhancedSentimentAnalyzer.load_model(model_path)
    print("Model úspěšně načten")
    
    # Testovací texty
    test_texts = [
        "Toto je velmi dobrá zpráva, jsme nadšení z výsledků.",
        "Bohužel došlo k vážné nehodě a několik lidí bylo zraněno.",
        "Výsledky průzkumu ukazují, že 45% respondentů preferuje červenou barvu."
    ]
    
    # Testovat model na textech
    for text in test_texts:
        result = sentiment_model.predict([text])[0]
        print(f"Text: {text}")
        print(f"Předpověděný sentiment: {sentiment_model.labels[result]}")
        
        # Pokud model podporuje vysvětlení
        if hasattr(sentiment_model, 'explain_prediction'):
            explanation = sentiment_model.explain_prediction(text)
            print(f"Vysvětlení: {explanation}")
        print("-" * 50)
        
except Exception as e:
    print(f"Chyba při načítání nebo testování modelu: {str(e)}")