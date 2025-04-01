import sys
import os
import pickle

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
    
    # Upravit metodiku klasifikace sentimentu přímo v modelu
    def custom_predict(self, texts):
        """
        Vlastní funkce pro predikci sentimentu s upravenými prahovými hodnotami
        """
        # Extrahovat příznaky
        additional_features = self.extract_sentiment_features(texts)
        
        # Určit sentiment na základě ratio
        sentiments = []
        for i in range(len(texts)):
            ratio = additional_features['sentiment_ratio'].iloc[i]
            positive_count = additional_features['positive_word_count'].iloc[i]
            negative_count = additional_features['negative_word_count'].iloc[i]
            
            # Upravené prahové hodnoty
            if positive_count > 0 and (ratio > 1.5 or positive_count >= 2 and negative_count == 0):
                # Pozitivní sentiment při výrazně více pozitivních slovech
                sentiments.append(2)  # positive
            elif negative_count > 0 and (ratio < 0.7 or negative_count >= 2 and positive_count == 0):
                # Negativní sentiment při výrazně více negativních slovech
                sentiments.append(0)  # negative
            else:
                # Neutrální sentiment pro vyvážené nebo žádné emoční výrazy
                sentiments.append(1)  # neutral
        
        return sentiments
    
    # Nahradit existující metodu predict
    EnhancedSentimentAnalyzer.predict = custom_predict
    
    # Přiřadit novou metodu ke konkrétní instanci
    sentiment_model.predict = custom_predict.__get__(sentiment_model, EnhancedSentimentAnalyzer)
    
    # Uložit upravený model
    sentiment_model.save_model(model_path)
    print("Model s upravenou klasifikační funkcí úspěšně uložen")
    
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