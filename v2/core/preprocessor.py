"""Text preprocessing: tokenization and lemmatization"""
import re
from collections import Counter
import pymorphy3
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config


class Preprocessor:
    """Text preprocessing pipeline"""
    
    def __init__(self):
        # Russian morphology
        self.morph_ru = pymorphy3.MorphAnalyzer()
        
        # English lemmatizer
        self.lemmatizer_en = WordNetLemmatizer()
        
        # Stopwords
        try:
            self.stopwords_ru = set(stopwords.words('russian'))
            self.stopwords_en = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.stopwords_ru = set(stopwords.words('russian'))
            self.stopwords_en = set(stopwords.words('english'))
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words"""
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Split and lowercase
        tokens = text.lower().split()
        
        # Filter by length
        tokens = [t for t in tokens if len(t) >= config.MIN_WORD_LENGTH]
        
        return tokens
    
    def lemmatize_ru(self, word: str) -> str:
        """Lemmatize Russian word"""
        parsed = self.morph_ru.parse(word)[0]
        return parsed.normal_form
    
    def lemmatize_en(self, word: str) -> str:
        """Lemmatize English word"""
        return self.lemmatizer_en.lemmatize(word)
    
    def lemmatize(self, word: str) -> str:
        """Lemmatize word (auto-detect language)"""
        # Try Russian first
        if any(ord(c) >= 1040 and ord(c) <= 1103 for c in word):
            return self.lemmatize_ru(word)
        else:
            return self.lemmatize_en(word)
    
    def is_stopword(self, word: str) -> bool:
        """Check if word is stopword"""
        return word in self.stopwords_ru or word in self.stopwords_en
    
    def process_text(self, text: str) -> tuple[list[str], list[str]]:
        """Process single text: tokenize and lemmatize"""
        tokens = self.tokenize(text)
        
        # Lemmatize and filter stopwords
        lemmas = []
        for token in tokens:
            if not self.is_stopword(token):
                lemma = self.lemmatize(token)
                if not self.is_stopword(lemma):
                    lemmas.append(lemma)
        
        return tokens, lemmas
    
    def process_texts(self, texts: list[str]) -> tuple[list[list[str]], list[list[str]]]:
        """Process multiple texts"""
        all_tokens = []
        all_lemmas = []
        
        for text in texts:
            tokens, lemmas = self.process_text(text)
            all_tokens.append(tokens)
            all_lemmas.append(lemmas)
        
        return all_tokens, all_lemmas
