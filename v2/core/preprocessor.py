"""Text preprocessing: tokenization and lemmatization."""
import re
import pymorphy3
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config


class Preprocessor:
    """Text preprocessing pipeline."""
    
    def __init__(self):
        self.morph_ru = pymorphy3.MorphAnalyzer()
        self.lemmatizer_en = WordNetLemmatizer()
        
        try:
            self.stopwords_ru = set(stopwords.words('russian'))
            self.stopwords_en = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self.stopwords_ru = set(stopwords.words('russian'))
            self.stopwords_en = set(stopwords.words('english'))
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = text.lower().split()
        return [t for t in tokens if len(t) >= config.MIN_WORD_LENGTH]
    
    def lemmatize_ru(self, word: str) -> str:
        """Lemmatize Russian word."""
        return self.morph_ru.parse(word)[0].normal_form
    
    def lemmatize_en(self, word: str) -> str:
        """Lemmatize English word."""
        return self.lemmatizer_en.lemmatize(word)
    
    def lemmatize(self, word: str) -> str:
        """Lemmatize word (auto-detect language)."""
        if any(1040 <= ord(c) <= 1103 for c in word):
            return self.lemmatize_ru(word)
        return self.lemmatize_en(word)
    
    def is_stopword(self, word: str) -> bool:
        """Check if word is stopword."""
        return word in self.stopwords_ru or word in self.stopwords_en
    
    def process_text(self, text: str) -> tuple[list[str], list[str]]:
        """Process single text: tokenize and lemmatize."""
        tokens = self.tokenize(text)
        
        lemmas = []
        for token in tokens:
            if not self.is_stopword(token):
                lemma = self.lemmatize(token)
                if not self.is_stopword(lemma):
                    lemmas.append(lemma)
        
        return tokens, lemmas
    
    def process_texts(self, texts: list[str]) -> tuple[list[list[str]], list[list[str]]]:
        """Process multiple texts."""
        all_tokens = []
        all_lemmas = []
        
        for text in texts:
            tokens, lemmas = self.process_text(text)
            all_tokens.append(tokens)
            all_lemmas.append(lemmas)
        
        return all_tokens, all_lemmas
