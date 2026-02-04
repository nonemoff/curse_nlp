# Data Files Documentation

This directory contains configuration files for NLP analysis.

## Files

### domain_terms.txt
Dictionary of domain-specific terms (319 terms).
- Format: one term per line (Russian or English)
- Categories: distributed systems, consensus algorithms, blockchain, NLP
- Used to boost TF-IDF scores for relevant terms

### abbr_stopwords.txt
Stop words for abbreviation filtering.
- Excludes: technical labels (URL, DOI), organizations (IEEE), roman numerals (XVI)

### location_abbr.txt
Location abbreviation mapping.
- Format: `abbr=full_name`
- Example: `СПб=Санкт-Петербург`

### journal_markers.txt
Keywords for identifying journal/conference titles.
- Used to filter out publications from organization entities

### address_markers.txt
Keywords for identifying address components.
- Used to filter out street names from location entities

### products.txt
List of software products and technologies.
- Blockchain platforms: Bitcoin, Ethereum, etc.
- Consensus algorithms: PoW, PoS, PBFT, etc.

## Editing

To modify:
1. Edit the `.txt` file directly
2. Run `python main.py clear` to clear cache
3. Run `python main.py all` to reanalyze with new settings
