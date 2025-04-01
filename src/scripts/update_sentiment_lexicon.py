import sys
import os

# Přidání správných cest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer

# Cesta k modelu
model_path = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')

try:
    # Načíst model
    sentiment_model = EnhancedSentimentAnalyzer.load_model(model_path)
    print("Model úspěšně načten")
    
    # Původní lexikony
    print(f"Původní počet pozitivních slov: {len(sentiment_model.positive_words)}")
    print(f"Původní počet negativních slov: {len(sentiment_model.negative_words)}")
    
    # Rozšířený seznam českých pozitivních slov
    positive_words = [
        'dobrý', 'skvělý', 'výborný', 'pozitivní', 'úspěch', 'radost', 'krásný', 'příjemný',
        'štěstí', 'spokojený', 'výhra', 'zisk', 'růst', 'lepší', 'nejlepší', 'zlepšení',
        'výhoda', 'prospěch', 'podpora', 'rozvoj', 'pokrok', 'úspěšný', 'optimistický',
        'šťastný', 'veselý', 'bezpečný', 'klidný', 'prospěšný', 'úžasný', 'perfektní',
        'vynikající', 'senzační', 'fantastický', 'neuvěřitelný', 'báječný', 'nádherný',
        'velkolepý', 'luxusní', 'přátelský', 'laskavý', 'milý', 'ochotný', 'talentovaný',
        'nadaný', 'inovativní', 'kreativní', 'silný', 'výkonný', 'efektivní', 'užitečný',
        'cenný', 'důležitý', 'ohromující', 'fascinující', 'zajímavý', 'pozoruhodný',
        'inspirativní', 'motivující', 'povzbuzující', 'osvěžující', 'uvolňující',
        'uklidňující', 'příznivý', 'konstruktivní', 'produktivní', 'perspektivní',
        'slibný', 'nadějný', 'obohacující', 'vzrušující', 'úchvatný', 'impozantní', 
        'působivý', 'přesvědčivý', 'vítaný', 'populární', 'oblíbený', 'milovaný',
        'oceňovaný', 'oslavovaný', 'vyzdvihovaný', 'vyžadovaný', 'potřebný', 'žádoucí',
        'velmi', 'skvěle', 'nadšení', 'nadšený', 'radostný', 'vylepšený', 'přelomový',
        'úžasně', 'nadmíru', 'mimořádně', 'výjimečně', 'srdečně', 'ideální', 'dobře'
    ]
    
    # Rozšířený seznam českých negativních slov
    negative_words = [
        'špatný', 'negativní', 'problém', 'potíž', 'selhání', 'prohra', 'ztráta', 'pokles',
        'krize', 'konflikt', 'smrt', 'válka', 'nehoda', 'tragédie', 'nebezpečí', 'zhoršení',
        'škoda', 'nízký', 'horší', 'nejhorší', 'slabý', 'nepříznivý', 'riziko', 'hrozba',
        'kritický', 'závažný', 'obtížný', 'těžký', 'násilí', 'strach', 'obavy', 'útok',
        'katastrofa', 'pohroma', 'neštěstí', 'destrukce', 'zničení', 'zkáza', 'porážka',
        'kolaps', 'pád', 'děsivý', 'hrozný', 'strašný', 'příšerný', 'otřesný', 'hrozivý',
        'znepokojivý', 'alarmující', 'ohavný', 'odpudivý', 'nechutný', 'odporný', 'krutý',
        'brutální', 'agresivní', 'surový', 'barbarský', 'divoký', 'vražedný', 'smrtící',
        'jedovatý', 'toxický', 'škodlivý', 'ničivý', 'zničující', 'fatální', 'smrtelný',
        'zoufalý', 'beznadějný', 'bezmocný', 'deprimující', 'skličující', 'depresivní',
        'smutný', 'bolestný', 'trýznivý', 'traumatický', 'poškozený', 'rozbitý', 'zlomený',
        'naštvaný', 'rozzlobený', 'rozzuřený', 'rozhořčený', 'nenávistný', 'nepřátelský',
        'odmítavý', 'podvodný', 'klamavý', 'lživý', 'falešný', 'neetický', 'nemorální',
        'zkorumpovaný', 'zkažený', 'prohnilý', 'bezcenný', 'zbytečný', 'marný', 'bídný',
        'ubohý', 'žalostný', 'nedostatečný', 'průměrný', 'nudný', 'nezajímavý', 'nezáživný',
        'bohužel', 'žel', 'naneštěstí', 'nešťastný', 'narušený', 'znechucený', 'zraněný',
        'zraněno', 'utrpení', 'trápení', 'vážné', 'vážně', 'kriticky', 'drasticky', 'hrozně',
        'selhal', 'selhala', 'nepovedlo', 'nefunguje', 'chyba', 'nefunkční', 'rozpadlý'
    ]
    
    # Aktualizace lexikonů v modelu
    sentiment_model.positive_words = list(set(positive_words))  # Odstranění duplicit
    sentiment_model.negative_words = list(set(negative_words))  # Odstranění duplicit
    
    print(f"Nový počet pozitivních slov: {len(sentiment_model.positive_words)}")
    print(f"Nový počet negativních slov: {len(sentiment_model.negative_words)}")
    
    # Uložení aktualizovaného modelu
    sentiment_model.save_model(model_path)
    print("Model s aktualizovanými lexikony úspěšně uložen")
    
    # Test s příklady
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
        
        # Vysvětlení
        explanation = sentiment_model.explain_prediction(text)
        print(f"Vysvětlení: {explanation}")
        print("-" * 50)
    
except Exception as e:
    print(f"Chyba: {str(e)}")