import sys
import os

# Přidání správných cest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from models.enhanced_category_classifier import EnhancedCategoryClassifier
from data.text_preprocessor import TextPreprocessor

# Cesta k modelu
model_path = os.path.join(project_root, 'models', 'enhanced_category_classifier')

try:
    # Načíst model
    category_model = EnhancedCategoryClassifier.load_model(model_path)
    text_preprocessor = TextPreprocessor(language='czech')
    print("Model úspěšně načten")
    
    # Testovací text - článek z ukázky
    test_text = """
    Bez komentáře: Ukrajinci shodili klouzavé bomby na velitelství, které řídilo operace přes Dněpr.
    Video: telegramsoniah. Útok ukazuje pokračující bitvu o strategicky výhodná místa v Chersonské oblasti. Ta je pro Ukrajinu důležitá, protože se nachází u ústí Dněpru a zajišťuje zemi přístup k Černému moři. Ukrajinská armáda 
    bombardovala velitelský bunkr ruské armády v okupované části Chersonské oblasti. Záběry z místa útoku sdílel 
    například investigativní účet OSINTtechnical, spravovaný analytikern z amerického think-tanku Centrum námořní 
    analýzy (CNA), o útoku informoval také server časopisu Forbes. Jejich autenticitu ale nelze nezávisle ověřit, stejně 
    jako veškeré detaily o okolnostech útoku. Ukrajinci na bunkr údajně zaútočili pomocí klouzavé bomby GBU-62 
    řízené americkým naváděcím systémem Joint Direct Attack Munition (JDAM). Ukrajinci od USA dostala i verzi JDAM-
    ER. Jde o typ s prodlouženým doletem a přidavnými křídly. V podzemí u obce Oleška se mělo nacházet ruské 
    velitelské stanoviště pro danou oblast, uvádí účet NOELreports, který sdružuje analytiky a válku na Ukrajině poměrně 
    společlivě monitoruje. Podle Forbesu ruské síly využily nefunkční velitelský bunkr protivzdušné obrany — kdysi 
    spojený s baterií raket S-300 ukrajinského letectva. Mohl se jim jevit jako bezpečné místo pro polní velitelství, 
    protože se nacházel v podzemí a vstup byl řádně opevněn. Cílem úderu měla být likvidace velitelského sboru 
    odpovědného za útoky na ostrovy na řece Dněpr, o které se obě strany dlouhodobé přetahují. Žádné vedení — žádné 
    vylodění na našich ostrovech, komentoval útok jeden z ukrajinských vojenských blogerů. Klouzavé bomby na 
    Ukrajině hojně používají obě válčící strany. Jde o přesně naváděné letecké pumy s dosahem až několik desítek 
    kilometrů — pokud jsou shozené z dostatečné výšky. Obecným pravidlem je, že čím vyšší výška, tím delší dolet. Cíl 
    mohou zasáhnout s odchylkou sotva několika metrů, jelikož jsou vybaveny navigačními systémy a aerodynamickými 
    křídly.
    """
    
    # Předzpracování textu
    processed_text = text_preprocessor.preprocess_text(test_text)
    
    # Predikce kategorie
    predicted_category = category_model.predict([processed_text])[0]
    print(f"Předpověděná kategorie: {predicted_category}")
    
    # Analýza politického zaměření
    def analyze_political_bias(text):
        # Keywords associated with different political orientations in Czech context
        left_wing_terms = ['sociální', 'odbory', 'rovnost', 'spravedlnost', 'práva', 'pracující', 'solidarita']
        right_wing_terms = ['trh', 'podnikání', 'ekonomický', 'svoboda', 'daně', 'snížení', 'rozpočet']
        pro_ukraine_terms = ['ukrajinská armáda', 'ukrajinci', 'ukrajinského', 'naši']
        pro_russia_terms = ['ruské síly', 'ruská armáda', 'osvobození']
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count occurrences of terms
        left_count = sum(text_lower.count(term) for term in left_wing_terms)
        right_count = sum(text_lower.count(term) for term in right_wing_terms)
        ukraine_count = sum(text_lower.count(term) for term in pro_ukraine_terms)
        russia_count = sum(text_lower.count(term) for term in pro_russia_terms)
        
        # Determine political bias
        political_score = left_count - right_count
        conflict_bias_score = ukraine_count - russia_count
        
        # Political bias result
        if political_score > 1:
            political_bias = "levicově orientovaný"
        elif political_score < -1:
            political_bias = "pravicově orientovaný"
        else:
            political_bias = "politicky neutrální"
            
        # Conflict bias result
        if conflict_bias_score > 1:
            conflict_bias = "nakloněný Ukrajině"
        elif conflict_bias_score < -1:
            conflict_bias = "nakloněný Rusku"
        else:
            conflict_bias = "neutrální k oběma stranám konfliktu"
            
        return {
            "political_bias": political_bias,
            "conflict_bias": conflict_bias,
            "left_count": left_count,
            "right_count": right_count,
            "ukraine_count": ukraine_count,
            "russia_count": russia_count
        }
    
    # Analyzujeme politické zaměření textu
    political_analysis = analyze_political_bias(test_text)
    print(f"Politické zaměření: {political_analysis['political_bias']}")
    print(f"Zaměření v konfliktu: {political_analysis['conflict_bias']}")
    print(f"Levicové výrazy: {political_analysis['left_count']}")
    print(f"Pravicové výrazy: {political_analysis['right_count']}")
    print(f"Pro-Ukrajina výrazy: {political_analysis['ukraine_count']}")
    print(f"Pro-Rusko výrazy: {political_analysis['russia_count']}")
    
    # Teď musíme upravit funkce v chatbot.py
    
except Exception as e:
    print(f"Chyba: {str(e)}")