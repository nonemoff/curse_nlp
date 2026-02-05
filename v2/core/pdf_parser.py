"""PDF text extraction module."""
import fitz
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm


class PDFParser:
    """Extract text from PDF files."""
    
    def __init__(self, corpus_dir: Path, output_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text(self, pdf_path: Path) -> dict:
        """Extract text from single PDF file."""
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        
        try:
            lang = detect(text[:1000])
            lang = 'RU' if lang == 'ru' else 'EN'
        except LangDetectException:
            lang = 'UNKNOWN'
        
        output_file = self.output_dir / f"{pdf_path.stem}.txt"
        output_file.write_text(text, encoding='utf-8')
        
        return {
            'filename': pdf_path.name,
            'text': text,
            'char_count': len(text),
            'language': lang,
            'output_file': str(output_file)
        }
    
    def extract_all(self) -> list[dict]:
        """Extract text from all PDFs in corpus directory."""
        pdf_files = list(self.corpus_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.corpus_dir}")
            return []
        
        results = []
        for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
            try:
                result = self.extract_text(pdf_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
        
        return results
