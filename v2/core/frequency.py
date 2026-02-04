"""Frequency analysis module"""
import math
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import config


class FrequencyAnalyzer:
    """Frequency analysis of lemmatized texts"""
    
    def __init__(self):
        self.results = None
    
    def analyze(self, lemma_lists: list[list[str]]) -> dict:
        """Perform frequency analysis"""
        # Flatten all lemmas
        all_lemmas = [lemma for lemmas in lemma_lists for lemma in lemmas]
        
        # Count frequencies
        freq_counter = Counter(all_lemmas)
        
        # Sort by frequency
        sorted_lemmas = freq_counter.most_common()
        
        # Calculate statistics
        M = len(all_lemmas)  # Total tokens
        N = len(freq_counter)  # Unique lemmas
        K_R = (N / M) * 100 if M > 0 else 0  # Diversity coefficient
        K_I = M / N if N > 0 else 0  # Informativity coefficient
        
        # Build frequency dictionary
        freq_dict = []
        cumulative = 0
        for rank, (lemma, freq) in enumerate(sorted_lemmas, 1):
            cumulative += freq
            freq_dict.append({
                'lemma': lemma,
                'frequency': freq,
                'rank': rank,
                'relative_freq': (freq / M) * 100,
                'cumulative_freq': cumulative
            })
        
        # Find core lexicon (50% coverage)
        core_size = 0
        for item in freq_dict:
            if item['cumulative_freq'] >= M * config.CORE_LEXICON_THRESHOLD:
                core_size = item['rank']
                break
        
        self.results = {
            'M': M,
            'N': N,
            'K_R': K_R,
            'K_I': K_I,
            'core_lexicon_size': core_size,
            'freq_dict': freq_dict
        }
        
        return self.results
    
    def save_results(self, output_dir: Path):
        """Save results to files"""
        if not self.results:
            return
        
        output_dir = Path(output_dir)
        
        # Save frequency dictionary
        df = pd.DataFrame(self.results['freq_dict'])
        df.to_csv(output_dir / 'frequency_dict.csv', index=False, sep=';', encoding='utf-8-sig')
        
        # Save statistics
        stats_file = output_dir / 'statistics.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"Частотный анализ\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"M (объём текста): {self.results['M']:,}\n")
            f.write(f"N (объём словаря): {self.results['N']:,}\n")
            f.write(f"K_R (разнообразие): {self.results['K_R']:.2f}\n")
            f.write(f"K_I (информативность): {self.results['K_I']:.2f}\n")
            f.write(f"Ядро лексики (50%): {self.results['core_lexicon_size']} слов\n")
    
    def plot_zipf(self, output_path: Path):
        """Plot Zipf's law"""
        if not self.results:
            return
        
        freq_dict = self.results['freq_dict'][:1000]  # Top 1000
        ranks = [item['rank'] for item in freq_dict]
        freqs = [item['frequency'] for item in freq_dict]
        
        plt.figure(figsize=config.GRAPH_FIGSIZE)
        plt.loglog(ranks, freqs, 'b.', alpha=0.5, label='Фактическое')
        
        # Theoretical Zipf line
        C = freqs[0]
        theoretical = [C / r for r in ranks]
        plt.loglog(ranks, theoretical, 'r--', label='Теоретическое (закон Ципфа)')
        
        plt.xlabel('Ранг (log)')
        plt.ylabel('Частота (log)')
        plt.title('Закон Ципфа')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=config.GRAPH_DPI)
        plt.close()
    
    def plot_cumulative(self, output_path: Path):
        """Plot cumulative frequency"""
        if not self.results:
            return
        
        freq_dict = self.results['freq_dict']
        ranks = [item['rank'] for item in freq_dict]
        cumulative = [item['cumulative_freq'] for item in freq_dict]
        
        M = self.results['M']
        core_size = self.results['core_lexicon_size']
        
        plt.figure(figsize=config.GRAPH_FIGSIZE)
        plt.plot(ranks, cumulative, 'b-', linewidth=2)
        
        # 50% line
        plt.axhline(y=M * 0.5, color='r', linestyle='--', label='50% текста')
        plt.axvline(x=core_size, color='k', linestyle='--', label=f'Ядро: {core_size} слов')
        
        plt.xlabel('Ранг слова')
        plt.ylabel('Накопленная частота')
        plt.title('Динамика накопления лексики')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=config.GRAPH_DPI)
        plt.close()
