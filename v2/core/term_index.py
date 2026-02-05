"""Terminological index builder with TF-IDF."""
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import pymorphy3

import config


class TermIndexBuilder:
    """Build terminological index using TF-IDF."""
    
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        
        self.domain_terms = set()
        domain_file = config.DATA_FILES['domain_terms']
        if domain_file.exists():
            self.domain_terms = set(config.load_text_file(domain_file))
        
        self.abbr_stopwords = set()
        abbr_file = config.DATA_FILES['abbr_stopwords']
        if abbr_file.exists():
            self.abbr_stopwords = set(config.load_text_file(abbr_file))
        
        self.results = None
    
    def is_valid_term(self, word: str) -> bool:
        """Check if word is valid term (noun or adjective)."""
        parsed = self.morph.parse(word)[0]
        return 'NOUN' in parsed.tag or 'ADJF' in parsed.tag
    
    def extract_terms(self, texts: list[str]) -> list[dict]:
        """Extract single-word terms with TF-IDF."""
        doc_words = []
        for text in texts:
            words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
            doc_words.append(words)
        
        df = defaultdict(int)
        for words in doc_words:
            unique_words = set(words)
            for word in unique_words:
                df[word] += 1
        
        num_docs = len(texts)
        term_scores = {}
        
        for words in doc_words:
            word_freq = Counter(words)
            for word, freq in word_freq.items():
                if not self.is_valid_term(word):
                    continue
                
                tf = freq / len(words) if words else 0
                idf = math.log(num_docs / df[word]) + 1 if df[word] > 0 else 0
                tfidf = tf * idf
                
                if word in self.domain_terms:
                    tfidf *= config.DOMAIN_BOOST
                
                if word not in term_scores or tfidf > term_scores[word]:
                    term_scores[word] = tfidf
        
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'term': word, 'tfidf_score': score, 'in_domain': word in self.domain_terms}
            for word, score in sorted_terms if score > 0
        ]
    
    def extract_ngrams(self, texts: list[str], n: int) -> list[dict]:
        """Extract n-grams with TF-IDF."""
        doc_ngrams = []
        for text in texts:
            words = re.findall(r'\b[а-яёa-z]+\b', text.lower())
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            doc_ngrams.append(ngrams)
        
        df = defaultdict(int)
        for ngrams in doc_ngrams:
            unique_ngrams = set(ngrams)
            for ngram in unique_ngrams:
                df[ngram] += 1
        
        num_docs = len(texts)
        ngram_scores = {}
        
        for ngrams in doc_ngrams:
            ngram_freq = Counter(ngrams)
            for ngram, freq in ngram_freq.items():
                if freq < config.MIN_TERM_FREQUENCY:
                    continue
                
                tf = freq / len(ngrams) if ngrams else 0
                idf = math.log(num_docs / df[ngram]) + 1 if df[ngram] > 0 else 0
                tfidf = tf * idf
                
                if ngram in self.domain_terms:
                    tfidf *= config.DOMAIN_BOOST
                
                if ngram not in ngram_scores or tfidf > ngram_scores[ngram]:
                    ngram_scores[ngram] = tfidf
        
        sorted_ngrams = sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'term': ngram, 'tfidf_score': score, 'in_domain': ngram in self.domain_terms}
            for ngram, score in sorted_ngrams if score > 0
        ]
    
    def extract_abbreviations(self, texts: list[str]) -> list[dict]:
        """Extract abbreviations with expansions."""
        ABBR_PATTERN = re.compile(r'\b([A-ZА-ЯЁ]{2,})\s*\(([^)]+)\)')
        REVERSE_PATTERN = re.compile(r'([^(]+)\s*\(([A-ZА-ЯЁ]{2,})\)')
        
        abbr_expansions = defaultdict(set)
        
        for text in texts:
            for match in ABBR_PATTERN.finditer(text):
                abbr = match.group(1)
                expansion = ' '.join(match.group(2).split())
                abbr_expansions[abbr].add(expansion)
            
            for match in REVERSE_PATTERN.finditer(text):
                expansion = ' '.join(match.group(1).strip().split())
                abbr = match.group(2)
                abbr_expansions[abbr].add(expansion)
        
        results = []
        var_pattern = re.compile(r'^[A-Z][0-9]$')
        
        for abbr, expansions in abbr_expansions.items():
            if abbr in self.abbr_stopwords or var_pattern.match(abbr):
                continue
            
            expansion = list(expansions)[0] if len(expansions) == 1 else '; '.join(expansions)
            
            results.append({
                'abbreviation': abbr,
                'expansion': expansion,
                'frequency': len(expansions)
            })
        
        results.sort(key=lambda x: x['frequency'], reverse=True)
        return results
    
    def build_index(self, texts: list[str]) -> dict:
        """Build complete terminological index."""
        terms = self.extract_terms(texts)
        bigrams = self.extract_ngrams(texts, 2)
        trigrams = self.extract_ngrams(texts, 3)
        abbreviations = self.extract_abbreviations(texts)
        
        all_terms = terms + bigrams + trigrams
        
        domain_terms = [t for t in all_terms if t['in_domain']]
        other_terms = [t for t in all_terms if not t['in_domain']]
        
        domain_terms.sort(key=lambda x: x['tfidf_score'], reverse=True)
        other_terms.sort(key=lambda x: x['tfidf_score'], reverse=True)
        
        diversified = domain_terms + other_terms
        
        self.results = {
            'terms': terms,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'abbreviations': abbreviations,
            'all_terms': diversified[:config.TOP_WORDS_LIMIT],
            'total': len(terms) + len(bigrams) + len(trigrams) + len(abbreviations),
            'domain_count': sum(1 for t in all_terms if t['in_domain'])
        }
        
        return self.results
    
    def save_results(self, output_dir: Path):
        """Save results to CSV files."""
        if not self.results:
            return
        
        output_dir = Path(output_dir)
        
        df_terms = pd.DataFrame(self.results['all_terms'][:100])
        df_terms.to_csv(output_dir / 'term_index.csv', index=False, sep=';', encoding='utf-8-sig')
        
        df_abbr = pd.DataFrame(self.results['abbreviations'])
        if 'expansion' in df_abbr.columns:
            df_abbr['expansion'] = df_abbr['expansion'].apply(lambda x: ' '.join(str(x).split()))
        df_abbr.to_csv(output_dir / 'abbreviations.csv', index=False, sep=';', encoding='utf-8-sig')
