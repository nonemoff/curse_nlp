"""Terminological index builder with TF-IDF"""
import re
from collections import Counter, defaultdict
from pathlib import Path
import math
import pandas as pd
import pymorphy3

import config


class TermIndexBuilder:
    """Build terminological index using TF-IDF"""
    
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        
        # Load domain terms
        self.domain_terms = set()
        domain_file = config.DATA_FILES['domain_terms']
        if domain_file.exists():
            self.domain_terms = set(config.load_text_file(domain_file))
        
        # Load abbreviation stopwords
        self.abbr_stopwords = set()
        abbr_file = config.DATA_FILES['abbr_stopwords']
        if abbr_file.exists():
            self.abbr_stopwords = set(config.load_text_file(abbr_file))
        
        self.results = None
    
    def is_valid_term(self, word: str) -> bool:
        """Check if word is valid term (noun or adjective)"""
        parsed = self.morph.parse(word)[0]
        return 'NOUN' in parsed.tag or 'ADJF' in parsed.tag
    
    def extract_terms(self, texts: list[str]) -> dict:
        """Extract single-word terms with TF-IDF"""
        # Tokenize all texts
        doc_words = []
        for text in texts:
            words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
            doc_words.append(words)
        
        # Calculate document frequency
        df = defaultdict(int)
        for words in doc_words:
            unique_words = set(words)
            for word in unique_words:
                df[word] += 1
        
        # Calculate TF-IDF for each term
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
                
                # Boost domain terms
                if word in self.domain_terms:
                    tfidf *= config.DOMAIN_BOOST
                
                if word not in term_scores or tfidf > term_scores[word]:
                    term_scores[word] = tfidf
        
        # Sort by score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        terms = []
        for word, score in sorted_terms:
            if score > 0:
                terms.append({
                    'term': word,
                    'tfidf_score': score,
                    'in_domain': word in self.domain_terms
                })
        
        return terms
    
    def extract_ngrams(self, texts: list[str], n: int) -> list[dict]:
        """Extract n-grams with TF-IDF"""
        # Collect all n-grams from all documents
        doc_ngrams = []
        for text in texts:
            words = re.findall(r'\b[а-яёa-z]+\b', text.lower())
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            doc_ngrams.append(ngrams)
        
        # Calculate document frequency
        df = defaultdict(int)
        for ngrams in doc_ngrams:
            unique_ngrams = set(ngrams)
            for ngram in unique_ngrams:
                df[ngram] += 1
        
        # Calculate TF-IDF
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
                
                # Boost domain terms
                if ngram in self.domain_terms:
                    tfidf *= config.DOMAIN_BOOST
                
                if ngram not in ngram_scores or tfidf > ngram_scores[ngram]:
                    ngram_scores[ngram] = tfidf
        
        # Sort and format
        sorted_ngrams = sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for ngram, score in sorted_ngrams:
            if score > 0:
                results.append({
                    'term': ngram,
                    'tfidf_score': score,
                    'in_domain': ngram in self.domain_terms
                })
        
        return results
    
    def extract_abbreviations(self, texts: list[str]) -> list[dict]:
        """Extract abbreviations with expansions"""
        # Patterns
        ABBR_PATTERN = re.compile(r'\b([A-ZА-ЯЁ]{2,})\s*\(([^)]+)\)')
        REVERSE_PATTERN = re.compile(r'([^(]+)\s*\(([A-ZА-ЯЁ]{2,})\)')
        
        abbr_expansions = defaultdict(set)
        
        for text in texts:
            # Pattern 1: ABBR (expansion)
            for match in ABBR_PATTERN.finditer(text):
                abbr = match.group(1)
                expansion = ' '.join(match.group(2).split())  # Clean newlines
                abbr_expansions[abbr].add(expansion)
            
            # Pattern 2: expansion (ABBR)
            for match in REVERSE_PATTERN.finditer(text):
                expansion = ' '.join(match.group(1).strip().split())
                abbr = match.group(2)
                abbr_expansions[abbr].add(expansion)
        
        # Filter and format
        results = []
        var_pattern = re.compile(r'^[A-Z][0-9]$')
        
        for abbr, expansions in abbr_expansions.items():
            # Filter stopwords
            if abbr in self.abbr_stopwords:
                continue
            
            # Filter single-letter variables (P1, T2, etc.)
            if var_pattern.match(abbr):
                continue
            
            expansion = list(expansions)[0] if len(expansions) == 1 else '; '.join(expansions)
            
            results.append({
                'abbreviation': abbr,
                'expansion': expansion,
                'frequency': len(expansions)
            })
        
        # Sort by frequency
        results.sort(key=lambda x: x['frequency'], reverse=True)
        
        return results
    
    def build_index(self, texts: list[str]) -> dict:
        """Build complete terminological index"""
        # Extract all types
        terms = self.extract_terms(texts)
        bigrams = self.extract_ngrams(texts, 2)
        trigrams = self.extract_ngrams(texts, 3)
        abbreviations = self.extract_abbreviations(texts)
        
        # Diversify top terms (alternate between 1/2/3-word terms)
        all_terms = terms + bigrams + trigrams
        
        # Separate domain and non-domain
        domain_terms = [t for t in all_terms if t['in_domain']]
        other_terms = [t for t in all_terms if not t['in_domain']]
        
        # Sort each by score
        domain_terms.sort(key=lambda x: x['tfidf_score'], reverse=True)
        other_terms.sort(key=lambda x: x['tfidf_score'], reverse=True)
        
        # Merge: domain first, then others
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
        """Save results to CSV files"""
        if not self.results:
            return
        
        output_dir = Path(output_dir)
        
        # Save terms
        df_terms = pd.DataFrame(self.results['all_terms'][:100])  # Top 100
        df_terms.to_csv(output_dir / 'term_index.csv', index=False, sep=';', encoding='utf-8-sig')
        
        # Save abbreviations
        df_abbr = pd.DataFrame(self.results['abbreviations'])
        # Clean newlines in expansions
        if 'expansion' in df_abbr.columns:
            df_abbr['expansion'] = df_abbr['expansion'].apply(lambda x: ' '.join(str(x).split()))
        df_abbr.to_csv(output_dir / 'abbreviations.csv', index=False, sep=';', encoding='utf-8-sig')
