"""Configuration settings for NLP project"""
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent
CORPUS_DIR = ROOT_DIR.parent / "corpus"
OUTPUT_DIR = ROOT_DIR / "output"
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR = ROOT_DIR / "data"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "extracted").mkdir(exist_ok=True)
(OUTPUT_DIR / "graphs").mkdir(exist_ok=True)

# Data files
DATA_FILES = {
    'domain_terms': DATA_DIR / 'domain_terms.txt',
    'products': DATA_DIR / 'products.txt',
    'abbr_stopwords': DATA_DIR / 'abbr_stopwords.txt',
    'location_abbr': DATA_DIR / 'location_abbr.txt',
    'journal_markers': DATA_DIR / 'journal_markers.txt',
    'address_markers': DATA_DIR / 'address_markers.txt',
}

# Frequency analysis settings
MIN_WORD_LENGTH = 3
TOP_WORDS_LIMIT = 1000
CORE_LEXICON_THRESHOLD = 0.5

# Term extraction settings
MIN_TERM_FREQUENCY = 3
MAX_NGRAM_LENGTH = 3
MIN_TERMS_REQUIRED = 100
DOMAIN_BOOST = 2.0

# NER settings
MIN_ENTITY_FREQUENCY = 2
MIN_PERSON_FREQUENCY = 1  # Lower threshold for persons

# Graph settings
GRAPH_DPI = 300
GRAPH_FIGSIZE = (12, 8)


def load_text_file(filepath: Path, skip_comments=True) -> list[str]:
    """Load lines from text file, optionally skipping comments"""
    if not filepath.exists():
        return []
    
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if skip_comments and (not line or line.startswith('#')):
                continue
            lines.append(line)
    return lines


def load_abbr_map(filepath: Path) -> dict[str, str]:
    """Load abbreviation mapping from key=value file"""
    if not filepath.exists():
        return {}
    
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                mapping[key.strip()] = value.strip()
    return mapping
