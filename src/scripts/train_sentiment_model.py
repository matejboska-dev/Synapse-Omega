import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# TensorFlow a Keras pro neuronovou síť
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Scikit-learn pro metriky a dělení dat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_sentiment_model_neural_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# absolute imports from src directory
from models.sentiment_analyzer import SentimentAnalyzer

# logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Definice nové třídy neuronového analyzátoru sentimentu
class NeuralSentimentAnalyzer:
    """
    Sentiment Analyzer založený na neuronové síti LSTM
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=500):
        """
        Inicializace analyzátoru sentimentu s neuronovou sítí
        
        Args:
            vocab_size (int): Velikost slovníku (počet nejčastějších slov)
            embedding_dim (int): Dimenze vektorů slov (embedding)
            max_length (int): Maximální délka článku (počet slov)
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.model = None
        self.labels = ['negative', 'neutral', 'positive']
        self.history = None
        
        # Načtení českých pozitivních a negativních slov pro vysvětlení
        self.positive_words = [
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
        
        self.negative_words = [
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
    
    def build_model(self):
        """
        Vytvoření modelu neuronové sítě
        """
        model = Sequential([
            # Embedding vrstva převádí tokeny na vektory
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            
            # Bidirectional LSTM pro zachycení kontextu v obou směrech
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            
            # Druhá LSTM vrstva
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            
            # Plně propojené vrstvy
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            # Výstupní vrstva - 3 třídy: negativní, neutrální, pozitivní
            Dense(3, activation='softmax')
        ])
        
        # Kompilace modelu
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        
        self.model = model
        return model
    
    def fit(self, texts, labels, validation_split=0.2, epochs=15, batch_size=64):
        """
        Trénování modelu neuronové sítě
        
        Args:
            texts (list): Seznam textů pro trénování
            labels (list): Seznam labelů (0=negativní, 1=neutrální, 2=pozitivní)
            validation_split (float): Část dat pro validaci
            epochs (int): Počet epoch tréninku
            batch_size (int): Velikost dávky
            
        Returns:
            dict: Výsledky tréninku
        """
        # Tokenizace textů
        logger.info("Tokenizace textů...")
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, truncating='post', padding='post')
        
        # Dělení dat na trénovací a validační
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, np.array(labels),
            test_size=validation_split,
            random_state=42,
            stratify=labels
        )
        
        # Vytvoření modelu, pokud ještě neexistuje
        if self.model is None:
            self.build_model()
            logger.info(f"Model vytvořen s architekturou:")
            self.model.summary(print_fn=lambda x: logger.info(x))
        
        # Callbacks pro zlepšení tréninku
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001),
            ModelCheckpoint(
                filepath='best_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        # Trénink modelu
        logger.info(f"Začátek tréninku na {len(X_train)} vzorcích, validace na {len(X_val)} vzorcích...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Načtení nejlepšího modelu
        if os.path.exists('best_model.h5'):
            self.model = tf.keras.models.load_model('best_model.h5')
            logger.info("Načten nejlepší model podle přesnosti validace")
        
        # Vyhodnocení modelu
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Predikce pro další metriky
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        confusion = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=self.labels, output_dict=True)
        
        if os.path.exists('best_model.h5'):
            os.remove('best_model.h5')  # Odstraníme dočasný soubor
        
        # Detailní výsledky
        results = {
            'accuracy': val_acc,
            'loss': val_loss,
            'confusion_matrix': confusion.tolist(),
            'classification_report': report,
            'training_history': {
                'accuracy': self.history.history['accuracy'],
                'val_accuracy': self.history.history['val_accuracy'],
                'loss': self.history.history['loss'],
                'val_loss': self.history.history['val_loss']
            }
        }
        
        return results
    
    def predict(self, texts):
        """
        Predikce sentimentu pro dané texty
        
        Args:
            texts (list): Seznam textů k analýze
            
        Returns:
            list: Seznam predikovaných tříd (0=negativní, 1=neutrální, 2=pozitivní)
        """
        if self.model is None:
            raise ValueError("Model není natrénován. Nejprve zavolejte metodu fit().")
        
        # Tokenizace a padding textů
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, truncating='post', padding='post')
        
        # Predikce
        predictions = self.model.predict(padded_sequences)
        return np.argmax(predictions, axis=1).tolist()
    
    def predict_proba(self, texts):
        """
        Predikce pravděpodobností sentimentu pro dané texty
        
        Args:
            texts (list): Seznam textů k analýze
            
        Returns:
            list: Seznam vektorů pravděpodobností tříd
        """
        if self.model is None:
            raise ValueError("Model není natrénován. Nejprve zavolejte metodu fit().")
        
        # Tokenizace a padding textů
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, truncating='post', padding='post')
        
        # Predikce pravděpodobností
        return self.model.predict(padded_sequences).tolist()
    
    def extract_sentiment_features(self, texts):
        """
        Extrakce příznaků souvisejících se sentimentem (pro kompatibilitu s předchozí verzí)
        
        Args:
            texts (list): Seznam textů k analýze
            
        Returns:
            pd.DataFrame: DataFrame s extrahovanými příznaky
        """
        features = pd.DataFrame()
        
        # Počítání pozitivních a negativních slov
        features['positive_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.positive_words) 
            for text in tqdm(texts, desc="Counting positive words")
        ]
        
        features['negative_word_count'] = [
            sum(1 for word in text.lower().split() if word in self.negative_words)
            for text in tqdm(texts, desc="Counting negative words")
        ]
        
        # Sentiment ratio (positive vs negative)
        features['sentiment_ratio'] = (features['positive_word_count'] + 1) / (features['negative_word_count'] + 1)
        
        # Text length features
        features['text_length'] = [len(text) for text in texts]
        features['word_count'] = [len(text.split()) for text in texts]
        
        return features
    
    def explain_prediction(self, text):
        """
        Vysvětlení predikce sentimentu pro daný text
        
        Args:
            text (str): Text k analýze
            
        Returns:
            dict: Vysvětlení predikce
        """
        # Počítání pozitivních a negativních slov
        positive_words_found = [word for word in text.lower().split() if word in self.positive_words]
        negative_words_found = [word for word in text.lower().split() if word in self.negative_words]
        
        positive_word_count = len(positive_words_found)
        negative_word_count = len(negative_words_found)
        
        # Výpočet poměru sentimentu
        sentiment_ratio = (positive_word_count + 1) / (negative_word_count + 1)
        
        # Predikce
        sentiment_id = self.predict([text])[0]
        sentiment = self.labels[sentiment_id]
        
        # Confidence - pravděpodobnost predikce
        proba = self.predict_proba([text])[0]
        confidence = proba[sentiment_id]
        
        # Věty v textu
        sentences = text.split('.')
        
        # Analyzovat sentiment jednotlivých vět
        sentence_sentiments = []
        if len(sentences) > 1:
            for sentence in sentences:
                if len(sentence.strip()) > 5:  # Pouze smysluplné věty
                    sent_id = self.predict([sentence])[0]
                    sentence_sentiments.append((sentence, self.labels[sent_id]))
        
        # Vytvoření vysvětlení
        if sentiment == 'positive':
            if positive_word_count > 0:
                reason = f"Text obsahuje pozitivní slova jako: {', '.join(positive_words_found[:5])}"
            else:
                reason = "Text má celkově pozitivní tón, i když neobsahuje konkrétní pozitivní slova z našeho slovníku."
        elif sentiment == 'negative':
            if negative_word_count > 0:
                reason = f"Text obsahuje negativní slova jako: {', '.join(negative_words_found[:5])}"
            else:
                reason = "Text má celkově negativní tón, i když neobsahuje konkrétní negativní slova z našeho slovníku."
        else:
            reason = "Text obsahuje vyváženou směs pozitivních a negativních slov nebo neobsahuje dostatek slov s emočním nábojem."
        
        # Určení sentiment skóre
        sentiment_score = (positive_word_count - negative_word_count) / max(len(text.split()), 1) * 10
        
        return {
            'text': text,
            'predicted_sentiment': sentiment,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'positive_words': positive_words_found[:10],
            'negative_words': negative_words_found[:10],
            'positive_word_count': positive_word_count,
            'negative_word_count': negative_word_count,
            'word_count': len(text.split()),
            'sentiment_ratio': sentiment_ratio,
            'reason': reason,
            'sentence_analysis': sentence_sentiments[:5]  # Omezení na prvních 5 vět
        }
    
    def plot_training_history(self, figsize=(12, 5)):
        """
        Vykreslení historie tréninku
        
        Args:
            figsize (tuple): Velikost grafu
        """
        if self.history is None:
            raise ValueError("Historie tréninku není k dispozici. Nejprve zavolejte metodu fit().")
        
        plt.figure(figsize=figsize)
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        # Ensure figures directory exists
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        plt.savefig(os.path.join(figures_dir, 'neural_sentiment_training_history.png'))
        plt.close()
    
    def save_model(self, model_dir):
        """
        Uložení modelu na disk
        
        Args:
            model_dir (str): Cesta k adresáři pro uložení modelu
        """
        if self.model is None:
            raise ValueError("Model není natrénován. Nejprve zavolejte metodu fit().")
        
        # Vytvořit adresář, pokud neexistuje
        os.makedirs(model_dir, exist_ok=True)
        
        # Uložit model neuronové sítě
        tf_model_dir = os.path.join(model_dir, 'nn_model')
        if os.path.exists(tf_model_dir):
            import shutil
            shutil.rmtree(tf_model_dir)  # Odstranit existující adresář pro uložení modelu
            
        self.model.save(tf_model_dir)
        logger.info(f"Model uložen do {tf_model_dir}")
        
        # Uložit tokenizer
        with open(os.path.join(model_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Uložit lexikony
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'wb') as f:
            pickle.dump({
                'positive_words': self.positive_words,
                'negative_words': self.negative_words
            }, f)
        
        # Uložit model info
        model_info = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'labels': self.labels
        }
        
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Načtení modelu z disku
        
        Args:
            model_dir (str): Cesta k adresáři s uloženým modelem
            
        Returns:
            NeuralSentimentAnalyzer: Načtený model
        """
        # Načíst model info
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        # Vytvořit instanci
        instance = cls(
            vocab_size=model_info['vocab_size'],
            embedding_dim=model_info['embedding_dim'],
            max_length=model_info['max_length']
        )
        
        # Načíst toknenizer
        with open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb') as f:
            instance.tokenizer = pickle.load(f)
        
        # Načíst model neuronové sítě
        instance.model = tf.keras.models.load_model(os.path.join(model_dir, 'nn_model'))
        
        # Načíst lexikony
        with open(os.path.join(model_dir, 'lexicons.pkl'), 'rb') as f:
            lexicons = pickle.load(f)
            instance.positive_words = lexicons['positive_words']
            instance.negative_words = lexicons['negative_words']
        
        # Načíst labely
        instance.labels = model_info['labels']
        
        logger.info(f"Model načten z {model_dir}")
        return instance

# custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def plot_confusion_matrix(cm, class_names, figsize=(10, 8), cmap='Blues'):
    """
    plot confusion matrix
    
    args:
        cm (array): confusion matrix
        class_names (list): list of class names
        figsize (tuple): figure size
        cmap (str): colormap
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figures_dir, 'neural_sentiment_confusion_matrix.png'))
    plt.close()

def plot_class_distribution(class_dist, class_names, figsize=(8, 6)):
    """
    plot class distribution
    
    args:
        class_dist (array): class distribution
        class_names (list): list of class names
        figsize (tuple): figure size
    """
    plt.figure(figsize=figsize)
    plt.bar(class_names, class_dist)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    plt.tight_layout()
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figures_dir, 'neural_sentiment_distribution.png'))
    plt.close()

def create_balanced_dataset(df, column, min_samples=None, max_samples=None):
    """
    Create a balanced dataset with similar number of samples per class
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name with class labels
        min_samples (int): Minimum samples per class (if None, uses minimum class count)
        max_samples (int): Maximum samples per class (if None, uses minimum class count)
    
    Returns:
        pd.DataFrame: Balanced dataframe
    """
    class_counts = df[column].value_counts()
    min_class_count = class_counts.min()
    
    if min_samples is None:
        min_samples = min_class_count
    
    if max_samples is None:
        max_samples = min_class_count
    
    balanced_dfs = []
    
    for class_label, count in class_counts.items():
        class_df = df[df[column] == class_label]
        
        # If class has fewer samples than min_samples, oversample
        if count < min_samples:
            class_df = class_df.sample(min_samples, replace=True, random_state=42)
        # If class has more samples than max_samples, undersample
        elif count > max_samples:
            class_df = class_df.sample(max_samples, replace=False, random_state=42)
        
        balanced_dfs.append(class_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)

def main():
    """
    main function for training and evaluating the sentiment analyzer
    """
    # create output directories if they don't exist
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    model_dir = os.path.join(project_root, 'models', 'neural_sentiment_analyzer')
    for directory in [model_dir, 'reports/models', 'reports/figures']:
        dir_path = os.path.join(project_root, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"ensuring directory exists: {dir_path}")
    
    # load preprocessed data
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'articles_processed.csv')
    
    if not os.path.exists(processed_data_path):
        logger.error(f"Preprocessed data file not found: {processed_data_path}")
        logger.error("Run data_preparation.py first")
        return
    
    logger.info(f"Loading preprocessed data from {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    # check if required columns exist
    required_columns = ['Content', 'Title']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # basic info about data
    logger.info(f"Loaded {len(df)} articles")
    
    # Sample data if it's too large to prevent memory issues
    max_samples = 3000  # Maximum number of samples to use
    if len(df) > max_samples:
        logger.info(f"Dataset is large ({len(df)} samples), using a random sample of {max_samples}")
        df = df.sample(max_samples, random_state=42)
        logger.info(f"Using {len(df)} samples for training")
    
    # combine title and content for better analysis
    df['Text'] = df['Title'] + ' ' + df['Content']
    
    # manually create sentiment features for seeding the sentiment labels
    logger.info("Creating seed sentiment labels based on keyword analysis...")
    
    # Keywords for sentiment analysis
    positive_keywords = [
        'dobrý', 'skvělý', 'výborný', 'pozitivní', 'úspěch', 'radost', 'krásný', 'příjemný',
        'štěstí', 'spokojený', 'výhra', 'zisk', 'růst', 'lepší', 'nejlepší', 'zlepšení',
        'vynikající', 'fantastický', 'báječný', 'prospěšný', 'podpora', 'nadějný'
    ]
    
    negative_keywords = [
        'špatný', 'negativní', 'problém', 'potíž', 'selhání', 'prohra', 'ztráta', 'pokles',
        'krize', 'konflikt', 'smrt', 'válka', 'nehoda', 'tragédie', 'nebezpečí', 'zhoršení',
        'škoda', 'horší', 'nejhorší', 'slabý', 'riziko', 'hrozba', 'kritický', 'strach'
    ]
    
    # Create a function to assign sentiment based on keywords
    def assign_seed_sentiment(text):
        text = text.lower()
        pos_count = sum(1 for word in positive_keywords if word in text)
        neg_count = sum(1 for word in negative_keywords if word in text)
        
        # Return sentiment class (0: negative, 1: neutral, 2: positive)
        if pos_count > neg_count + 1:
            return 2  # positive
        elif neg_count > pos_count + 1:
            return 0  # negative
        else:
            return 1  # neutral
    
    # Initialize seed_sentiment column with automatic sentiment analysis
    logger.info("Assigning initial sentiment labels...")
    df['seed_sentiment'] = df['Text'].apply(assign_seed_sentiment)
    
    # Check distribution of seed sentiments
    seed_distribution = df['seed_sentiment'].value_counts()
    logger.info(f"Initial sentiment distribution: {seed_distribution.to_dict()}")
    
    # Balance the dataset to improve training
    logger.info("Balancing dataset...")
    balanced_df = create_balanced_dataset(df, 'seed_sentiment', 
                                         min_samples=min(500, seed_distribution.min()),
                                         max_samples=1000)
    
    # Check balanced distribution
    balanced_distribution = balanced_df['seed_sentiment'].value_counts()
    logger.info(f"Balanced sentiment distribution: {balanced_distribution.to_dict()}")
    
    # train neural sentiment model
    logger.info("Training neural sentiment model...")
    
    # Initialize model
    neural_analyzer = NeuralSentimentAnalyzer(
        vocab_size=20000,      # Slovník 20K nejčastějších slov
        embedding_dim=200,     # Dimenze embeddings
        max_length=500         # Maximální délka článku v tokenech
    )
    
    # Train model
    train_data = balanced_df['Text'].values
    train_labels = balanced_df['seed_sentiment'].values
    
    logger.info(f"Training neural sentiment model on {len(train_data)} samples...")
    
    # Train with progress bar
    results = neural_analyzer.fit(
        train_data, 
        train_labels,
        validation_split=0.2,
        epochs=20,          # Maximum epochs
        batch_size=32      # Batch size
    )
    
    # Plot training history
    neural_analyzer.plot_training_history()
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        neural_analyzer.labels,
        figsize=(10, 8)
    )
    
    # Plot class distribution
    plot_class_distribution(
        balanced_distribution.values,
        neural_analyzer.labels,
        figsize=(8, 6)
    )
    
    # Save evaluation results
    results_path = os.path.join(
        project_root, 'reports', 'models', f'neural_sentiment_analyzer_results.json'
    )
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    # Save model
    neural_analyzer.save_model(model_dir)
    logger.info(f"Neural sentiment model saved to {model_dir}")
    
    # Create main model directory
    main_model_dir = os.path.join(project_root, 'models', 'enhanced_sentiment_analyzer')
    os.makedirs(main_model_dir, exist_ok=True)
    
    # Copy files from neural model to main directory for compatibility
    try:
        # Create symlink or copy files
        for subdir, dirs, files in os.walk(model_dir):
            for file in files:
                src_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(src_path, model_dir)
                dst_path = os.path.join(main_model_dir, rel_path)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy file
                import shutil
                shutil.copy2(src_path, dst_path)
        
        # Copy model_info.pkl with adjusted settings
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        
        model_info['model_type'] = 'neural'  # Add model type
        
        with open(os.path.join(main_model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
        
        logger.info(f"Model files copied to main directory {main_model_dir}")
    except Exception as e:
        logger.error(f"Error copying model files: {str(e)}")
    
    logger.info(f"Training completed. Results saved to reports/models/")
    
    # test sentiment analyzer on a few sample articles
    logger.info("Testing neural sentiment analyzer on sample articles...")
    
    # load trained model
    analyzer = NeuralSentimentAnalyzer.load_model(model_dir)
    
    # sample articles
    sample_idx = np.random.randint(0, len(df), size=5)
    samples = df.iloc[sample_idx]
    
    for _, article in samples.iterrows():
        text = article['Text']
        sentiment_id = analyzer.predict([text])[0]
        
        logger.info(f"Title: {article['Title'][:50]}...")
        logger.info(f"Predicted sentiment: {analyzer.labels[sentiment_id]}")
        
        # Get detailed explanation
        explanation = analyzer.explain_prediction(text)
        positive_words = explanation['positive_words']
        negative_words = explanation['negative_words']
        ratio = explanation['sentiment_ratio']
        
        logger.info(f"Positive words: {positive_words}")
        logger.info(f"Negative words: {negative_words}")
        logger.info(f"Confidence: {explanation['confidence']:.2f}")
        logger.info(f"Ratio: {ratio:.2f}")
        logger.info(f"Reason: {explanation['reason']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    try:
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU is available: {gpus}")
            # Set memory growth to avoid memory allocation errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.info("No GPU found, using CPU for training")
        
        # Run main function
        main()
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")