"""Named Entity Recognition (NER) module"""
import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

# Russian NER
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Doc
)

# English NER
import spacy
import pymorphy3

import config


class NERExtractor:
    """Extract named entities from texts"""
    
    def __init__(self):
        # Russian NER (Natasha)
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        
        # English NER (spaCy)
        try:
            self.nlp_en = spacy.load('en_core_web_sm')
        except:
            print("Warning: spacy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp_en = None
        
        # Pymorphy for validation
        self.morph = pymorphy3.MorphAnalyzer()
        
        # Load configuration files
        self.location_abbr_map = config.load_abbr_map(config.DATA_FILES['location_abbr'])
        self.journal_markers = set(config.load_text_file(config.DATA_FILES['journal_markers']))
        self.address_markers = set(config.load_text_file(config.DATA_FILES['address_markers']))
        
        # Load products list
        self.products = set()
        products_file = config.DATA_FILES['products']
        if products_file.exists():
            self.products = set(config.load_text_file(products_file))
        
        self.results = None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize person/location name"""
        # Remove extra spaces
        name = ' '.join(name.split())
        
        # Check abbreviation map
        if name in self.location_abbr_map:
            return self.location_abbr_map[name]
        
        return name
    
    def _normalize_location(self, name: str) -> str:
        """Normalize location to nominative case"""
        words = name.split()
        
        if len(words) > 1:
            # Inflect first word to nominative
            parsed = self.morph.parse(words[0])[0]
            if 'nomn' not in parsed.tag:
                inflected = parsed.inflect({'nomn'})
                if inflected:
                    words[0] = inflected.word.capitalize()
            name = ' '.join(words)
        
        return self._normalize_name(name)
    
    def _validate_person(self, name: str) -> bool:
        """Validate person name"""
        # Must have at least 2 chars
        if len(name) < 2:
            return False
        
        # Check for name patterns
        patterns = [
            r'^[А-ЯЁA-Z][а-яёa-z]+\s+[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]\.',  # Иванов И.И.
            r'^[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]\.\s+[А-ЯЁA-Z][а-яёa-z]+',  # И.И. Иванов
            r'^[А-ЯЁA-Z][а-яёa-z]+\s+[А-ЯЁA-Z][а-яёa-z]+',        # John Smith
        ]
        
        for pattern in patterns:
            if re.match(pattern, name):
                return True
        
        # Exclude bibliographic terms
        exclude_terms = ['parallel distrib', 'et al', 'proc', 'ieee']
        if any(term in name.lower() for term in exclude_terms):
            return False
        
        # Check with pymorphy
        words = name.split()
        if len(words) >= 1:
            parsed = self.morph.parse(words[0])[0]
            if 'Name' in parsed.tag or 'Surn' in parsed.tag or 'Patr' in parsed.tag:
                return True
        
        return False
    
    def _validate_org(self, name: str) -> bool:
        """Validate organization name"""
        if len(name) < 3:
            return False
        
        # Must contain org markers
        org_markers = ['университет', 'институт', 'university', 'company', 'institute', 'corporation']
        has_marker = any(marker in name.lower() for marker in org_markers)
        
        # Exclude journals
        if any(marker in name.lower() for marker in self.journal_markers):
            return False
        
        return has_marker
    
    def _validate_location(self, name: str) -> bool:
        """Validate location name"""
        if len(name) < 3:
            return False
        
        # Exclude address markers
        if any(marker in name.lower() for marker in self.address_markers):
            return False
        
        return True
    
    def _extract_entities_ru(self, text: str) -> dict:
        """Extract entities from Russian text"""
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)
        
        entities = defaultdict(list)
        
        for span in doc.spans:
            if span.type == 'PER':
                name = self._normalize_name(span.text)
                if self._validate_person(name):
                    entities['Персоналии'].append(name)
            
            elif span.type == 'ORG':
                name = self._normalize_name(span.text)
                if self._validate_org(name):
                    entities['Организации'].append(name)
            
            elif span.type == 'LOC':
                name = self._normalize_location(span.text)
                if self._validate_location(name):
                    entities['Топонимы'].append(name)
        
        return entities
    
    def _extract_entities_en(self, text: str) -> dict:
        """Extract entities from English text"""
        if not self.nlp_en:
            return defaultdict(list)
        
        doc = self.nlp_en(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                name = self._normalize_name(ent.text)
                if self._validate_person(name):
                    entities['Персоналии'].append(name)
            
            elif ent.label_ == 'ORG':
                name = self._normalize_name(ent.text)
                if self._validate_org(name):
                    entities['Организации'].append(name)
            
            elif ent.label_ in ('GPE', 'LOC'):
                name = self._normalize_location(ent.text)
                if self._validate_location(name):
                    entities['Топонимы'].append(name)
            
            elif ent.label_ == 'PRODUCT':
                name = ent.text
                entities['Программные продукты'].append(name)
        
        return entities
    
    def _extract_products(self, text: str) -> list[str]:
        """Extract software products from text"""
        found = []
        for product in self.products:
            pattern = r'\b' + re.escape(product) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found.append(product)
        return found
    
    def extract_from_corpus(self, corpus: list[dict]) -> dict:
        """Extract entities from entire corpus"""
        all_entities = defaultdict(list)
        
        for item in corpus:
            text = item['text']
            lang = item.get('language', 'EN')
            
            # Extract based on language
            if lang == 'RU':
                entities = self._extract_entities_ru(text)
            else:
                entities = self._extract_entities_en(text)
            
            # Merge
            for category, names in entities.items():
                all_entities[category].extend(names)
            
            # Extract products separately
            products = self._extract_products(text)
            all_entities['Программные продукты'].extend(products)
        
        # Count frequencies
        results_by_category = {}
        for category, names in all_entities.items():
            counter = Counter(names)
            
            # Apply frequency threshold
            min_freq = config.MIN_PERSON_FREQUENCY if category == 'Персоналии' else config.MIN_ENTITY_FREQUENCY
            
            filtered = {name: count for name, count in counter.items() if count >= min_freq}
            
            # Sort by frequency
            sorted_entities = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            
            results_by_category[category] = [
                {'name': name, 'frequency': count}
                for name, count in sorted_entities
            ]
        
        # Calculate totals
        total = sum(len(entities) for entities in results_by_category.values())
        
        self.results = {
            'by_category': results_by_category,
            'total': total
        }
        
        return self.results
    
    def save_results(self, output_dir: Path):
        """Save results to CSV"""
        if not self.results:
            return
        
        output_dir = Path(output_dir)
        
        # Combine all categories
        all_entities = []
        for category, entities in self.results['by_category'].items():
            for entity in entities:
                all_entities.append({
                    'name': entity['name'],
                    'category': category,
                    'frequency': entity['frequency']
                })
        
        # Save to CSV
        df = pd.DataFrame(all_entities)
        
        # Clean newlines
        if 'name' in df.columns:
            df['name'] = df['name'].apply(lambda x: ' '.join(str(x).split()))
        
        df.to_csv(output_dir / 'name_index.csv', index=False, sep=';', encoding='utf-8-sig')
