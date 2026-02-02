#!/usr/bin/env python3
"""
SPGC Corpus Filter for Non-Instrumental Text
=============================================

Filters the Standardized Project Gutenberg Corpus to extract poetry,
philosophy, religion, essays, and other non-instrumental writing while
excluding fiction, self-help, and instrumental prose.

Usage:
    python filter_spgc_corpus.py --spgc-dir /path/to/gutenberg --output-dir ./filtered_corpus

Prerequisites:
    1. Clone and run the SPGC pipeline:
       git clone https://github.com/pgcorpus/gutenberg.git
       cd gutenberg
       python get_data.py
    
    2. This creates:
       - metadata/metadata.csv
       - data/text/*.txt (cleaned text files)
       - data/raw/*.txt (raw files with headers/footers)
"""

import argparse
import csv
import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION: Adjust these lists to tune your corpus
# =============================================================================

# Bookshelf categories to INCLUDE (human-curated, lower overlap)
INCLUDE_BOOKSHELVES = [
    # Poetry
    'Poetry', 'Poetry, A-Z by first line', 'Harvard Classics',
    
    # Philosophy & Ethics  
    'Philosophy', 'Ethics', 'Stoicism',
    
    # Religion & Spirituality
    'Mysticism', 'Christianity', 'Buddhism', 'Hinduism',
    'Religious Society of Friends', 'Unitarianism',
    'Swedenborgianism', 'Theosophy',
    
    # Essays & Contemplative
    'Essays', 'Personal Narratives', 'Meditations',
    
    # Classics (often contemplative/philosophical)
    'Classics', 'Ancient World',
    
    # Nature writing (phenomenological, non-instrumental)
    'Natural History', 'Nature',
]

# Subject headings to INCLUDE (Library of Congress style)
INCLUDE_SUBJECTS = [
    # Poetry
    'Poetry', 'Poets', 'Verse',
    
    # Philosophy
    'Philosophy', 'Ethics', 'Metaphysics', 'Ontology',
    'Stoics', 'Epicureans', 'Platonists', 'Neoplatonism',
    
    # Religion & Spirituality
    'Religion', 'Spirituality', 'Mysticism', 'Meditation',
    'Theology', 'Contemplation', 'Devotional',
    'Bible', 'Upanishads', 'Vedas', 'Sutras',
    'Buddhism', 'Hinduism', 'Taoism', 'Sufism',
    'Monasticism', 'Asceticism', 'Hermits',
    'Sermons', 'Spiritual life',
    
    # Essays & Aphorisms
    'Essays', 'Aphorisms', 'Maxims', 'Meditations',
    
    # Classical literature
    'Classical literature', 'Greek literature', 'Latin literature',
    'Plato', 'Aristotle', 'Epictetus', 'Marcus Aurelius', 'Seneca',
    
    # Nature & Contemplative
    'Nature', 'Natural history',
]

# Subject headings to EXCLUDE (even if other criteria match)
EXCLUDE_SUBJECTS = [
    # Fiction (resolves plots, instrumental narrative)
    'Fiction', 'Novel', 'Romance', 'Short stories',
    'Science fiction', 'Fantasy', 'Mystery', 'Detective',
    'Adventure', 'Thriller', 'Horror', 'Western stories',
    'Love stories', 'Domestic fiction',
    
    # Children's literature
    'Juvenile', 'Children', 'Young adult', 'Camp Fire', 'Girl Scouts',  'Boy Scouts', 'Boarding school',
    
    # Instrumental/prescriptive
    'Self-help', 'How-to', 'Handbooks', 'Manuals',
    'Textbooks', 'Study guides', 'Examinations',
    'Cookery', 'Cooking', 'Recipes',
    'Business', 'Commerce', 'Economics',
    'Law', 'Legal',
    'Medicine', 'Medical', 'Health',
    'Technology', 'Engineering', 'Mechanics',
    
    # Periodicals & Reference
    'Periodicals', 'Magazines', 'Newspapers',
    'Encyclopedias', 'Dictionaries', 'Directories',
    
    # Political/argumentative (resolves toward positions)
    'Political science', 'Politics', 'Government',
    'Socialism', 'Communism', 'Anarchism',
    
    # Drama (mostly plot-driven)
    'Drama', 'Plays', 'Theater',
]

# Specific authors to prioritize (contemplative writers)
PRIORITY_AUTHORS = [
    # Western Mystics
    'Augustine', 'Meister Eckhart', 'John of the Cross',
    'Teresa of Avila', 'Julian of Norwich', 'Hildegard',
    'Thomas à Kempis', 'Brother Lawrence', 'Fenelon',
    'Jacob Boehme', 'Emanuel Swedenborg', 'William Law',
    
    # Philosophers
    'Plato', 'Plotinus', 'Marcus Aurelius', 'Epictetus',
    'Seneca', 'Boethius', 'Montaigne', 'Pascal',
    'Emerson', 'Thoreau', 'Nietzsche',
    
    # Poets
    'Rumi', 'Hafiz', 'Kabir', 'Blake', 'Whitman',
    'Rilke', 'Hopkins', 'Dickinson', 'Wordsworth',
    
    # Eastern (translations)
    'Lao Tzu', 'Chuang Tzu', 'Confucius',
]


# =============================================================================
# CORE FILTERING LOGIC
# =============================================================================

def matches_any(text: Optional[str], patterns: list[str], case_sensitive: bool = False) -> bool:
    """Check if text matches any pattern in the list."""
    if not text:
        return False
    if not case_sensitive:
        text = text.lower()
        patterns = [p.lower() for p in patterns]
    return any(pattern in text for pattern in patterns)


def is_priority_author(author: Optional[str]) -> bool:
    """Check if author is in priority list."""
    if not author:
        return False
    author_lower = author.lower()
    return any(pa.lower() in author_lower for pa in PRIORITY_AUTHORS)


def should_include(row: dict) -> tuple[bool, str]:
    """
    Determine if a book should be included in the filtered corpus.
    
    Returns:
        (include: bool, reason: str)
    """
    bookshelf = row.get('bookshelf', '') or ''
    subjects = row.get('subjects', '') or ''
    author = row.get('author', '') or ''
    title = row.get('title', '') or ''
    language = row.get('language', '') or ''
    
    # Only English texts
    if language and 'en' not in language.lower():
        return False, 'non-english'
    
    # Check exclusions first (strong filter)
    if matches_any(subjects, EXCLUDE_SUBJECTS):
        # Exception: priority authors bypass some exclusions
        if not is_priority_author(author):
            return False, 'excluded-subject'
    
    # Check for inclusion criteria
    reasons = []
    
    if matches_any(bookshelf, INCLUDE_BOOKSHELVES):
        reasons.append('bookshelf-match')
    
    if matches_any(subjects, INCLUDE_SUBJECTS):
        reasons.append('subject-match')
    
    if is_priority_author(author):
        reasons.append('priority-author')
    
    if reasons:
        return True, '+'.join(reasons)
    
    return False, 'no-match'


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (words * 1.3)."""
    words = len(text.split())
    return int(words * 1.3)


# =============================================================================
# FILE PROCESSING
# =============================================================================

def find_text_file(spgc_dir: Path, book_id: str) -> Optional[Path]:
    """
    Find the text file for a given book ID.
    
    SPGC structure varies; try multiple locations.
    """
    possible_paths = [
        spgc_dir / 'data' / 'text' / f'{book_id}_text.txt',
        spgc_dir / 'text' / f'{book_id}_text.txt',
        spgc_dir / 'data' / 'raw' / f'{book_id}_raw.txt',
        spgc_dir / 'raw' / f'{book_id}_raw.txt',
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def clean_gutenberg_text(text: str) -> str:
    """
    Remove Gutenberg headers/footers if present in raw files.
    """
    # Common start markers
    start_markers = [
        '*** START OF THIS PROJECT GUTENBERG',
        '*** START OF THE PROJECT GUTENBERG',
        '*END*THE SMALL PRINT',
    ]
    
    # Common end markers
    end_markers = [
        '*** END OF THIS PROJECT GUTENBERG',
        '*** END OF THE PROJECT GUTENBERG',
        'End of Project Gutenberg',
        'End of the Project Gutenberg',
    ]
    
    # Find start
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Move past the marker line
            newline_idx = text.find('\n', idx)
            if newline_idx != -1:
                start_idx = max(start_idx, newline_idx + 1)
    
    # Find end
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = min(end_idx, idx)
    
    return text[start_idx:end_idx].strip()


def process_book(
    row: dict,
    spgc_dir: Path,
    output_dir: Path,
    category_dirs: bool = True
) -> Optional[dict]:
    """
    Process a single book: find file, clean, copy to output.
    
    Returns metadata dict if successful, None otherwise.
    """
    book_id = row.get('id', row.get('guten_id', ''))
    if not book_id:
        return None
    
    # Find source file
    src_path = find_text_file(spgc_dir, str(book_id))
    if not src_path:
        logger.debug(f"Text file not found for {book_id}")
        return None
    
    try:
        text = src_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logger.warning(f"Error reading {src_path}: {e}")
        return None
    
    # Clean if raw file
    if '_raw' in src_path.name:
        text = clean_gutenberg_text(text)
    
    # Skip very short texts
    if len(text) < 1000:
        logger.debug(f"Skipping {book_id}: too short ({len(text)} chars)")
        return None
    
    # Determine output category
    included, reason = should_include(row)
    if not included:
        return None
    
    # Determine category for organization
    category = 'other'
    bookshelf = (row.get('bookshelf', '') or '').lower()
    subjects = (row.get('subjects', '') or '').lower()
    
    if 'poetry' in bookshelf or 'poetry' in subjects:
        category = 'poetry'
    elif 'philosophy' in bookshelf or 'philosophy' in subjects:
        category = 'philosophy'
    elif any(r in bookshelf or r in subjects for r in ['religion', 'mystic', 'spiritual', 'buddhis', 'hindu']):
        category = 'religion'
    elif 'essay' in bookshelf or 'essay' in subjects:
        category = 'essays'
    elif 'nature' in bookshelf or 'natural history' in subjects:
        category = 'nature'
    
    # Create output path
    if category_dirs:
        out_dir = output_dir / category
    else:
        out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Safe filename
    title = row.get('title', 'untitled')[:50].replace('/', '_').replace('\\', '_')
    author = row.get('author', 'unknown')[:30].replace('/', '_').replace('\\', '_')
    filename = f"{book_id}_{author}_{title}.txt".replace(' ', '_')
    filename = re.sub(r'[^\w\-_.]', '', filename)
    
    out_path = out_dir / filename
    out_path.write_text(text, encoding='utf-8')
    
    return {
        'id': book_id,
        'title': row.get('title', ''),
        'author': row.get('author', ''),
        'category': category,
        'reason': reason,
        'tokens': estimate_tokens(text),
        'chars': len(text),
        'path': str(out_path.relative_to(output_dir)),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def load_metadata(spgc_dir: Path) -> list[dict]:
    """Load SPGC metadata from CSV."""
    possible_paths = [
        spgc_dir / 'metadata' / 'metadata.csv',
        spgc_dir / 'metadata.csv',
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading metadata from {path}")
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                return list(reader)
    
    raise FileNotFoundError(f"Could not find metadata.csv in {spgc_dir}")


def filter_corpus(
    spgc_dir: Path,
    output_dir: Path,
    category_dirs: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Main filtering pipeline.
    
    Args:
        spgc_dir: Path to SPGC directory
        output_dir: Where to write filtered corpus
        category_dirs: Organize output by category subdirs
        dry_run: Just analyze, don't copy files
    
    Returns:
        Statistics dict
    """
    logger.info(f"Loading metadata from {spgc_dir}")
    metadata = load_metadata(spgc_dir)
    logger.info(f"Loaded {len(metadata)} books")
    
    # First pass: analyze what matches
    include_stats = Counter()
    exclude_stats = Counter()
    
    for row in metadata:
        included, reason = should_include(row)
        if included:
            include_stats[reason] += 1
        else:
            exclude_stats[reason] += 1
    
    logger.info(f"\nInclusion breakdown:")
    for reason, count in include_stats.most_common():
        logger.info(f"  {reason}: {count}")
    
    logger.info(f"\nExclusion breakdown:")
    for reason, count in exclude_stats.most_common(10):
        logger.info(f"  {reason}: {count}")
    
    total_to_include = sum(include_stats.values())
    logger.info(f"\nTotal to include: {total_to_include} books")
    
    if dry_run:
        return {
            'total_books': len(metadata),
            'included': total_to_include,
            'include_breakdown': dict(include_stats),
            'exclude_breakdown': dict(exclude_stats),
        }
    
    # Second pass: process files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed = []
    category_stats = Counter()
    total_tokens = 0
    
    for i, row in enumerate(metadata):
        included, _ = should_include(row)
        if not included:
            continue
        
        result = process_book(row, spgc_dir, output_dir, category_dirs)
        if result:
            processed.append(result)
            category_stats[result['category']] += 1
            total_tokens += result['tokens']
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(metadata)} books, {len(processed)} included")
    
    # Write manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_books': len(processed),
            'total_tokens': total_tokens,
            'category_breakdown': dict(category_stats),
            'books': processed,
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total books processed: {len(processed)}")
    logger.info(f"Estimated total tokens: {total_tokens:,}")
    logger.info(f"\nBy category:")
    for cat, count in category_stats.most_common():
        logger.info(f"  {cat}: {count}")
    logger.info(f"\nOutput written to: {output_dir}")
    logger.info(f"Manifest: {manifest_path}")
    
    return {
        'total_books': len(processed),
        'total_tokens': total_tokens,
        'category_breakdown': dict(category_stats),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Filter SPGC corpus for non-instrumental text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--spgc-dir', '-s',
        type=Path,
        required=True,
        help='Path to SPGC directory (containing metadata/ and data/)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('./filtered_corpus'),
        help='Output directory for filtered corpus'
    )
    parser.add_argument(
        '--flat',
        action='store_true',
        help='Output all files to single directory (no category subdirs)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Analyze only, do not copy files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    filter_corpus(
        spgc_dir=args.spgc_dir,
        output_dir=args.output_dir,
        category_dirs=not args.flat,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
