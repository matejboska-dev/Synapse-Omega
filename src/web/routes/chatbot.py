import json
import logging
import os
import sys
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, current_app
import random

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Imports from our modules
from models.enhanced_category_classifier import EnhancedCategoryClassifier
from models.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from data.text_preprocessor import TextPreprocessor

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create blueprint
chatbot_bp = Blueprint('chatbot', __name__)

# Initialize models
category_model = None
sentiment_model = None
text_preprocessor = None
article_stats = None

def load_models():
    """Load machine learning models"""
    global category_model, sentiment_model, text_preprocessor, article_stats
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Load category model
        category_model_path = os.path.join(project_root, 'models', 'enhanced_category_classifier')
        if os.path.exists(category_model_path):
            category_model = EnhancedCategoryClassifier.load_model(category_model_path)
            logger.info("Enhanced category classifier loaded successfully")
        else:
            logger.warning(f"Enhanced category model not found at {category_model_path}")
        
        # Load sentiment model
        sentiment_model_path = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')
        if os.path.exists(sentiment_model_path):
            sentiment_model = EnhancedSentimentAnalyzer.load_model(sentiment_model_path)
            logger.info("Enhanced sentiment analyzer loaded successfully")
        else:
            logger.warning(f"Enhanced sentiment model not found at {sentiment_model_path}")
        
        # Initialize text preprocessor
        text_preprocessor = TextPreprocessor(language='czech')
        
        # Compute article statistics
        compute_article_stats()
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

def compute_article_stats():
    """Compute statistics about articles for the chatbot to use"""
    global article_stats
    
    try:
        # Get application context
        if 'articles_df' in current_app.config and current_app.config['articles_df'] is not None:
            df = current_app.config['articles_df']
            
            article_stats = {
                'total_articles': len(df),
                'sources': df['Source'].nunique(),
                'categories': df['Category'].nunique(),
                'top_sources': df['Source'].value_counts().head(5).to_dict(),
                'top_categories': df['Category'].value_counts().head(5).to_dict(),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
                'avg_article_length': int(df['ArticleLength'].mean()) if 'ArticleLength' in df.columns else 0,
                'avg_word_count': int(df['WordCount'].mean()) if 'WordCount' in df.columns else 0,
            }
            
            # Additional stats if sentiment is available
            if 'sentiment' in df.columns:
                # Sources with most negative articles
                negative_by_source = df[df['sentiment'] == 'negative'].groupby('Source').size()
                article_stats['negative_sources'] = negative_by_source.sort_values(ascending=False).head(3).to_dict()
                
                # Sources with most positive articles
                positive_by_source = df[df['sentiment'] == 'positive'].groupby('Source').size()
                article_stats['positive_sources'] = positive_by_source.sort_values(ascending=False).head(3).to_dict()
            
            logger.info("Article statistics computed successfully")
        else:
            article_stats = {"error": "No article data available"}
            logger.warning("No article data available for computing statistics")
    
    except Exception as e:
        article_stats = {"error": str(e)}
        logger.error(f"Error computing article statistics: {str(e)}")

@chatbot_bp.route('/chatbot')
def chatbot_page():
    """Render chatbot page"""
    return render_template('chatbot.html')

@chatbot_bp.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """API endpoint for chatbot interactions"""
    if category_model is None or sentiment_model is None:
        load_models()
    
    # Get message from request
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'response': 'Prosím, napište nějakou zprávu.'})
    
    # Process the message
    response = process_message(message)
    
    return jsonify({'response': response})

def process_message(message):
    """Process user message and generate response"""
    message_lower = message.lower()
    
    # Check for sentiment explanation question
    if 'proč' in message_lower and ('negativní' in message_lower or 'pozitivní' in message_lower or 'neutrální' in message_lower):
        return explain_sentiment_classification(message)
    
    # Check for category explanation
    elif 'kategor' in message_lower and ('jak' in message_lower or 'proč' in message_lower):
        return explain_category_classification()
    
    # Check for sentiment algorithm question
    elif 'jak' in message_lower and 'sentiment' in message_lower:
        return explain_sentiment_algorithm()
    
    # Check for negative sources question
    elif 'negativ' in message_lower and 'zdroj' in message_lower:
        return get_negative_sources()
    
    # Check for positive sources question
    elif 'pozitiv' in message_lower and 'zdroj' in message_lower:
        return get_positive_sources()
    
    # Check for stats question
    elif 'statisti' in message_lower or 'kolik' in message_lower:
        return get_article_statistics(message)
    
    # Check if user wants to analyze text
    elif 'analyz' in message_lower or 'klasifik' in message_lower:
        return analyze_text(message)
    
    # Default response for unrecognized questions
    else:
        return generate_default_response(message)

def explain_sentiment_classification(message):
    """Explain why a text was classified with a particular sentiment"""
    try:
        # Extract the actual text to analyze
        if 'proč je' in message.lower():
            text_start = message.lower().find('proč je') + 7
            text = message[text_start:].strip()
            
            # Preprocess text
            processed_text = text_preprocessor.preprocess_text(text)
            
            # Get explanation
            explanation = sentiment_model.explain_prediction(processed_text)
            
            response = f"<p>Analyzoval jsem text: <em>\"{text}\"</em></p>"
            response += f"<p>Sentiment textu jsem vyhodnotil jako <strong>{explanation['predicted_sentiment']}</strong>.</p>"
            
            if explanation['predicted_sentiment'] == 'positive':
                response += "<p>Důvody pro pozitivní klasifikaci:</p><ul>"
                if explanation['positive_words']:
                    response += f"<li>Nalezl jsem pozitivní slova: <strong>{', '.join(explanation['positive_words'])}</strong></li>"
                response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
                response += "</ul>"
            
            elif explanation['predicted_sentiment'] == 'negative':
                response += "<p>Důvody pro negativní klasifikaci:</p><ul>"
                if explanation['negative_words']:
                    response += f"<li>Nalezl jsem negativní slova: <strong>{', '.join(explanation['negative_words'])}</strong></li>"
                response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
                response += "</ul>"
            
            else:
                response += "<p>Text byl klasifikován jako neutrální, protože:</p><ul>"
                response += f"<li>Počet pozitivních slov: <strong>{explanation['positive_word_count']}</strong></li>"
                response += f"<li>Počet negativních slov: <strong>{explanation['negative_word_count']}</strong></li>"
                response += f"<li>Tyto hodnoty jsou vyrovnané nebo příliš nízké pro jednoznačnou klasifikaci.</li>"
                response += "</ul>"
            
            return response
        else:
            return "Prosím, uveďte konkrétní text, který chcete analyzovat. Například: \"Proč je tento článek negativní?\" nebo \"Proč je zpráva o zvýšení daní negativní?\""
    
    except Exception as e:
        logger.error(f"Error explaining sentiment: {str(e)}")
        return "Omlouvám se, ale při analýze sentimentu došlo k chybě. Zkuste to prosím znovu s jiným textem."

def explain_category_classification():
    """Explain how category classification works"""
    response = """
    <p>Klasifikace kategorií článků funguje na základě strojového učení s využitím několika technik:</p>
    <ol>
        <li><strong>Předzpracování textu</strong> - odstraňuji diakritiku, stopslova (jako "a", "ale", "je") a převádím slova na základní tvary</li>
        <li><strong>Extrakce příznaků</strong> - vytvářím vektory příznaků z textu pomocí:
            <ul>
                <li>TF-IDF pro slova (jak důležité jsou slova v článku)</li>
                <li>N-gramy znaků (zachycují části slov a jejich kombinace)</li>
            </ul>
        </li>
        <li><strong>Trénování klasifikátoru</strong> - používám algoritmus, který se naučil vzorce z tisíců článků s již přiřazenými kategoriemi</li>
    </ol>
    
    <p>Při klasifikaci nového článku proces zahrnuje:</p>
    <ol>
        <li>Předzpracování textu článku stejným způsobem</li>
        <li>Extrakci stejných typů příznaků</li>
        <li>Použití natrénovaného modelu pro predikci nejpravděpodobnější kategorie</li>
    </ol>
    
    <p>Kategorie jsou určeny na základě specifických slov a frází, které model identifikoval jako charakteristické pro danou kategorii. Například články o sportu často obsahují slova jako "zápas", "hráč", "skóre", zatímco články o ekonomice typicky zmiňují "inflace", "trh", "investice" apod.</p>
    """
    return response

def explain_sentiment_algorithm():
    """Explain how sentiment analysis works"""
    response = """
    <p>Analýza sentimentu článků probíhá v několika krocích:</p>
    
    <ol>
        <li><strong>Lexikální analýza</strong> - používám rozsáhlý slovník pozitivních a negativních slov v češtině</li>
        <li><strong>Komplexní předzpracování textu</strong> - text očistím od diakritiky, odstraním stopslova a převedu na základní tvary</li>
        <li><strong>Extrakce příznaků</strong> - analyzuji:
            <ul>
                <li>Počet pozitivních a negativních slov</li>
                <li>Poměr pozitivních ku negativním slovům</li>
                <li>Speciální příznaky jako vykřičníky, otazníky, slova psaná velkými písmeny</li>
                <li>TF-IDF příznaky slov i znaků</li>
            </ul>
        </li>
        <li><strong>Klasifikace</strong> - kombinuji všechny tyto příznaky v modelu, který byl natrénován na velkém množství článků</li>
    </ol>
    
    <p>Výsledky klasifikace jsou ve třech kategoriích:</p>
    <ul>
        <li><strong>Pozitivní</strong> - článek obsahuje převážně pozitivní slova a fráze</li>
        <li><strong>Neutrální</strong> - článek je vyvážený nebo neobsahuje dostatek emočně zabarvených slov</li>
        <li><strong>Negativní</strong> - článek obsahuje převážně negativní slova a fráze</li>
    </ul>
    
    <p>Tento přístup dosahuje přesnosti přes 90% na testovacích datech, což je významné zlepšení oproti předchozím modelům.</p>
    """
    return response

def get_negative_sources():
    """Get information about sources with most negative articles"""
    if article_stats and 'negative_sources' in article_stats:
        response = "<p>Zdroje s nejvyšším počtem negativních článků:</p><ul>"
        
        for source, count in article_stats['negative_sources'].items():
            response += f"<li><strong>{source}</strong>: {count} negativních článků</li>"
        
        response += "</ul>"
        
        response += "<p>Je důležité poznamenat, že toto nemusí nutně znamenat, že tyto zdroje jsou více pesimistické - může to být ovlivněno i tím, o jakých tématech častěji informují.</p>"
        
        return response
    else:
        return "Omlouvám se, ale nemám dostupná data o negativních zdrojích. Je možné, že analýza sentimentu nebyla provedena nebo nemáme dostatek článků."

def get_positive_sources():
    """Get information about sources with most positive articles"""
    if article_stats and 'positive_sources' in article_stats:
        response = "<p>Zdroje s nejvyšším počtem pozitivních článků:</p><ul>"
        
        for source, count in article_stats['positive_sources'].items():
            response += f"<li><strong>{source}</strong>: {count} pozitivních článků</li>"
        
        response += "</ul>"
        
        response += "<p>Je dobré vědět, že tyto zdroje častěji publikují pozitivní zprávy, ale pamatujte, že to může souviset i s tématy, kterým se věnují.</p>"
        
        return response
    else:
        return "Omlouvám se, ale nemám dostupná data o pozitivních zdrojích. Je možné, že analýza sentimentu nebyla provedena nebo nemáme dostatek článků."

def get_article_statistics(message):
    """Get statistics about articles"""
    if not article_stats or 'error' in article_stats:
        return "Omlouvám se, ale nemám dostupná statistická data o článcích."
    
    message_lower = message.lower()
    
    # Check what kind of statistics the user wants
    if 'zdroj' in message_lower:
        response = "<p>Statistiky o zdrojích:</p><ul>"
        response += f"<li>Celkový počet zdrojů: <strong>{article_stats['sources']}</strong></li>"
        
        response += "<li>Nejčastější zdroje:</li><ul>"
        for source, count in article_stats['top_sources'].items():
            response += f"<li><strong>{source}</strong>: {count} článků</li>"
        response += "</ul>"
        
        response += "</ul>"
        
        return response
    
    elif 'kategor' in message_lower:
        response = "<p>Statistiky o kategoriích:</p><ul>"
        response += f"<li>Celkový počet kategorií: <strong>{article_stats['categories']}</strong></li>"
        
        response += "<li>Nejčastější kategorie:</li><ul>"
        for category, count in article_stats['top_categories'].items():
            response += f"<li><strong>{category}</strong>: {count} článků</li>"
        response += "</ul>"
        
        response += "</ul>"
        
        return response
    
    elif 'sentiment' in message_lower:
        if 'sentiment_distribution' in article_stats and article_stats['sentiment_distribution']:
            response = "<p>Distribuce sentimentu v článcích:</p><ul>"
            
            for sentiment, count in article_stats['sentiment_distribution'].items():
                percentage = count / article_stats['total_articles'] * 100
                response += f"<li><strong>{sentiment}</strong>: {count} článků ({percentage:.1f}%)</li>"
            
            response += "</ul>"
            
            return response
        else:
            return "Omlouvám se, ale nemám dostupná data o distribuci sentimentu článků."
    
    # Default to general statistics
    response = "<p>Obecné statistiky o článcích:</p><ul>"
    response += f"<li>Celkový počet článků: <strong>{article_stats['total_articles']}</strong></li>"
    response += f"<li>Počet zdrojů: <strong>{article_stats['sources']}</strong></li>"
    response += f"<li>Počet kategorií: <strong>{article_stats['categories']}</strong></li>"
    response += f"<li>Průměrná délka článku: <strong>{article_stats['avg_article_length']}</strong> znaků</li>"
    response += f"<li>Průměrný počet slov: <strong>{article_stats['avg_word_count']}</strong></li>"
    
    if 'sentiment_distribution' in article_stats and article_stats['sentiment_distribution']:
        response += "<li>Distribuce sentimentu:</li><ul>"
        for sentiment, count in article_stats['sentiment_distribution'].items():
            percentage = count / article_stats['total_articles'] * 100
            response += f"<li><strong>{sentiment}</strong>: {percentage:.1f}%</li>"
        response += "</ul>"
    
    response += "</ul>"
    
    return response

def analyze_text(message):
    """Analyze text provided by the user"""
    # Try to extract text to analyze
    text_to_analyze = ""
    
    if "analyzuj" in message.lower() or "klasifikuj" in message.lower():
        parts = message.split('"')
        if len(parts) >= 3:  # Text is enclosed in quotes
            text_to_analyze = parts[1]
        else:
            # Check if there's text after the keywords
            for keyword in ["analyzuj", "klasifikuj", "analýza", "klasifikace"]:
                if keyword in message.lower():
                    idx = message.lower().find(keyword) + len(keyword)
                    text_to_analyze = message[idx:].strip()
                    if text_to_analyze and text_to_analyze[0] in [',', ':', ' ']:
                        text_to_analyze = text_to_analyze[1:].strip()
                    break
    
    if not text_to_analyze:
        return "Abych mohl analyzovat text, prosím, napište jej ve vašem dotazu. Například: \"Analyzuj text: Vláda schválila nový zákon o daních.\" nebo \"Klasifikuj: Fotbalisté Sparty postoupili do finále.\""
    
    # Preprocess text
    processed_text = text_preprocessor.preprocess_text(text_to_analyze)
    
    # Get category prediction
    predicted_category = category_model.predict([processed_text])[0]
    
    # Get sentiment prediction
    sentiment_id = sentiment_model.predict([processed_text])[0]
    sentiment = sentiment_model.labels[sentiment_id]
    
    # Get sentiment explanation
    explanation = sentiment_model.explain_prediction(processed_text)
    
    # Create response
    response = f"<p>Analýza textu: <em>\"{text_to_analyze}\"</em></p>"
    
    response += "<p><strong>Výsledky:</strong></p>"
    response += f"<p>Kategorie: <strong>{predicted_category}</strong></p>"
    response += f"<p>Sentiment: <strong>{sentiment}</strong></p>"
    
    # Add explanation
    response += "<p><strong>Vysvětlení sentimentu:</strong></p>"
    
    if sentiment == 'positive':
        response += "<ul>"
        if explanation['positive_words']:
            response += f"<li>Text obsahuje pozitivní slova: <strong>{', '.join(explanation['positive_words'])}</strong></li>"
        response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
        response += "</ul>"
    
    elif sentiment == 'negative':
        response += "<ul>"
        if explanation['negative_words']:
            response += f"<li>Text obsahuje negativní slova: <strong>{', '.join(explanation['negative_words'])}</strong></li>"
        response += f"<li>Poměr pozitivních ku negativním slovům: <strong>{explanation['sentiment_ratio']:.2f}</strong></li>"
        response += "</ul>"
    
    else:  # neutral
        response += "<ul>"
        response += f"<li>Počet pozitivních slov: <strong>{explanation['positive_word_count']}</strong></li>"
        response += f"<li>Počet negativních slov: <strong>{explanation['negative_word_count']}</strong></li>"
        response += "<li>Poměr je vyrovnaný nebo text neobsahuje dostatek emočně zabarvených slov pro jednoznačnou klasifikaci.</li>"
        response += "</ul>"
    
    return response

def generate_default_response(message):
    """Generate default response for unrecognized questions"""
    default_responses = [
        "Můžete se mě zeptat na analýzu článků, vysvětlení sentimentu nebo kategorií, a také na statistiky o zpravodajských zdrojích.",
        "Nejsem si jistý, co máte na mysli. Zkuste se mě zeptat na analýzu textu, vysvětlení klasifikace nebo statistiky o článcích.",
        "Specializuji se na analýzu zpravodajských článků. Mohu vám pomoci s analýzou textu, vysvětlením sentimentu nebo informacemi o zdrojích.",
        "Nepochopil jsem váš dotaz. Mohu vám pomoci s analýzou sentimentu textu, kategorií článků nebo statistikami zdrojů."
    ]
    
    return random.choice(default_responses)

# Register blueprint in app.py with:
# from routes.chatbot import chatbot_bp
# app.register_blueprint(chatbot_bp)