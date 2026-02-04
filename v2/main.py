#!/usr/bin/env python3
"""
NLP Курсовая работа - CLI интерфейс
Анализ текста информационного ресурса
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from core.pdf_parser import PDFParser
from core.preprocessor import Preprocessor
from core.frequency import FrequencyAnalyzer
from core.term_index import TermIndexBuilder
from core.ner import NERExtractor
from core.cache import CacheManager

console = Console()


def cmd_extract(args):
    """Extract text from PDF files"""
    parser = PDFParser(config.CORPUS_DIR, config.OUTPUT_DIR / "extracted")
    
    console.print("[bold cyan]📄 Извлечение текста из PDF...[/bold cyan]")
    results = parser.extract_all()
    
    # Show statistics
    table = Table(title="Статистика извлечения")
    table.add_column("Файл", style="cyan")
    table.add_column("Язык", style="green")
    table.add_column("Знаков", justify="right", style="yellow")
    
    for result in results[:10]:  # Show first 10
        table.add_row(
            result['filename'],
            result['language'],
            f"{result['char_count']:,}"
        )
    
    console.print(table)
    console.print(f"\n[green]✓[/green] Извлечено {len(results)} файлов")
    
    # Save to cache
    cache = CacheManager()
    cache.save('extracted', results)


def cmd_analyze(args):
    """Perform frequency analysis"""
    console.print("[bold cyan]📊 Частотный анализ...[/bold cyan]")
    
    # Load extracted texts
    cache = CacheManager()
    extracted = cache.load('extracted')
    
    if not extracted:
        console.print("[red]Ошибка: сначала выполните extract[/red]")
        return
    
    # Preprocess
    preprocessor = Preprocessor()
    texts = [item['text'] for item in extracted]
    tokens, lemmas = preprocessor.process_texts(texts)
    
    # Analyze
    analyzer = FrequencyAnalyzer()
    results = analyzer.analyze(lemmas)
    
    # Save results
    analyzer.save_results(config.OUTPUT_DIR)
    analyzer.plot_zipf(config.OUTPUT_DIR / "graphs" / "zipf.png")
    analyzer.plot_cumulative(config.OUTPUT_DIR / "graphs" / "cumulative.png")
    
    cache.save('frequency', results)
    
    console.print(f"\n[green]✓[/green] M={results['M']:,}, N={results['N']:,}, K_R={results['K_R']:.2f}")


def cmd_terms(args):
    """Build terminological index"""
    console.print("[bold cyan]📚 Терминологический указатель...[/bold cyan]")
    
    cache = CacheManager()
    extracted = cache.load('extracted')
    
    if not extracted:
        console.print("[red]Ошибка: сначала выполните extract[/red]")
        return
    
    # Build index
    builder = TermIndexBuilder()
    texts = [item['text'] for item in extracted]
    results = builder.build_index(texts)
    
    # Save
    builder.save_results(config.OUTPUT_DIR)
    cache.save('terms', results)
    
    console.print(f"\n[green]✓[/green] Всего терминов: {results['total']}")
    console.print(f"  - Однословные: {len(results['terms'])}")
    console.print(f"  - 2-словные: {len(results['bigrams'])}")
    console.print(f"  - 3-словные: {len(results['trigrams'])}")
    console.print(f"  - Аббревиатуры: {len(results['abbreviations'])}")


def cmd_names(args):
    """Extract named entities"""
    console.print("[bold cyan]👤 Именной указатель...[/bold cyan]")
    
    cache = CacheManager()
    extracted = cache.load('extracted')
    
    if not extracted:
        console.print("[red]Ошибка: сначала выполните extract[/red]")
        return
    
    # Extract NER
    extractor = NERExtractor()
    results = extractor.extract_from_corpus(extracted)
    
    # Save
    extractor.save_results(config.OUTPUT_DIR)
    cache.save('names', results)
    
    console.print(f"\n[green]✓[/green] Всего сущностей: {results['total']}")
    for category, entities in results['by_category'].items():
        console.print(f"  - {category}: {len(entities)}")


def cmd_all(args):
    """Run full pipeline"""
    cmd_extract(args)
    cmd_analyze(args)
    cmd_terms(args)
    cmd_names(args)
    console.print("\n[bold green]✓ Полный анализ завершён![/bold green]")


def cmd_status(args):
    """Show cache status"""
    cache = CacheManager()
    status = cache.get_status()
    
    table = Table(title="Статус кеша")
    table.add_column("Модуль", style="cyan")
    table.add_column("Статус", style="green")
    table.add_column("Размер", justify="right")
    
    for module, info in status.items():
        table.add_row(
            module,
            "✓ Есть" if info['exists'] else "✗ Нет",
            info.get('size', '-')
        )
    
    console.print(table)


def cmd_clear(args):
    """Clear cache"""
    cache = CacheManager()
    cache.clear_all()
    console.print("[green]✓ Кеш очищен[/green]")


def main():
    parser = argparse.ArgumentParser(description="NLP Курсовая работа")
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Commands
    subparsers.add_parser('extract', help='Извлечение текста из PDF')
    subparsers.add_parser('analyze', help='Частотный анализ')
    subparsers.add_parser('terms', help='Терминологический указатель')
    subparsers.add_parser('names', help='Именной указатель')
    subparsers.add_parser('all', help='Полный пайплайн')
    subparsers.add_parser('status', help='Статус кеша')
    subparsers.add_parser('clear', help='Очистить кеш')
    
    # Parse
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Dispatch
    commands = {
        'extract': cmd_extract,
        'analyze': cmd_analyze,
        'terms': cmd_terms,
        'names': cmd_names,
        'all': cmd_all,
        'status': cmd_status,
        'clear': cmd_clear,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
