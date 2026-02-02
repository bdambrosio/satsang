#!/usr/bin/env python3
"""
PDF to Text Converter using GROBID
==================================

Parse a PDF file using GROBID and extract text to an output file.

Usage:
    python pdf-to-text.py --input document.pdf --output document.txt
    python pdf-to-text.py --input document.pdf --output document.txt --grobid-url http://localhost:8070

Requirements:
    pip install requests
    GROBID server running (default: http://localhost:8070)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_pdf_grobid(pdf_filepath: str, title: str = None, grobid_url: str = "http://localhost:8070") -> dict:
    """
    Parse PDF using GROBID server.
    
    Args:
        pdf_filepath: Path to PDF file
        title: Optional title for the document
        grobid_url: GROBID server URL (default: http://localhost:8070)
    
    Returns:
        Dictionary with parsed content:
        {
            "title": str,
            "authors": str,
            "abstract": str,
            "chunks": [(section_title, section_text), ...]
        }
    """
    import requests
    
    if not os.path.exists(pdf_filepath):
        raise FileNotFoundError(f"PDF file not found: {pdf_filepath}")
    
    # Extract title from filename if not provided
    if title is None:
        title = Path(pdf_filepath).stem
    
    # GROBID endpoint for processing PDF
    process_url = f"{grobid_url}/api/processFulltextDocument"
    
    try:
        with open(pdf_filepath, 'rb') as pdf_file:
            files = {'input': pdf_file}
            data = {
                'generateIDs': '1',
                'consolidateCitations': '0',
                'includeRawCitations': '0',
                'includeRawAffiliations': '0',
                'teiCoordinates': '0'
            }
            
            logger.info(f"Sending PDF to GROBID server at {grobid_url}...")
            response = requests.post(process_url, files=files, data=data, timeout=300)
            response.raise_for_status()
            
            # Parse TEI XML response
            tei_xml = response.text
            result = _parse_tei_xml(tei_xml, title)
            
            logger.info(f"Successfully parsed PDF: {len(result.get('chunks', []))} sections")
            return result
            
    except requests.exceptions.RequestException as e:
        logger.error(f"GROBID request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"GROBID parsing failed: {e}")
        raise


def _parse_tei_xml(tei_xml: str, title: str = None) -> dict:
    """
    Parse GROBID TEI XML output into structured format.
    
    Args:
        tei_xml: TEI XML string from GROBID
        title: Optional title override
    
    Returns:
        Dictionary with parsed content
    """
    try:
        from xml.etree import ElementTree as ET
    except ImportError:
        logger.error("xml.etree.ElementTree not available")
        raise ImportError("xml.etree.ElementTree is required")
    
    try:
        root = ET.fromstring(tei_xml)
        
        # Namespace handling
        namespaces = {
            'tei': 'http://www.tei-c.org/ns/1.0',
            'default': 'http://www.tei-c.org/ns/1.0'
        }
        
        # Extract title
        title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', namespaces)
        if title_elem is None:
            title_elem = root.find('.//tei:titleStmt/tei:title', namespaces)
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
        
        # Extract authors
        authors = []
        for author in root.findall('.//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author', namespaces):
            pers_name = author.find('tei:persName', namespaces)
            if pers_name is not None:
                first = pers_name.find('tei:forename', namespaces)
                last = pers_name.find('tei:surname', namespaces)
                if first is not None and last is not None:
                    authors.append(f"{first.text} {last.text}".strip())
                elif last is not None:
                    authors.append(last.text.strip())
        authors_str = ", ".join(authors) if authors else ""
        
        # Extract abstract
        abstract_elem = root.find('.//tei:profileDesc/tei:abstract', namespaces)
        abstract = ""
        if abstract_elem is not None:
            abstract_parts = []
            for p in abstract_elem.findall('.//tei:p', namespaces):
                if p.text:
                    abstract_parts.append(p.text.strip())
            abstract = "\n".join(abstract_parts)
        
        # Extract sections/chunks
        chunks = []
        body = root.find('.//tei:text/tei:body', namespaces)
        if body is not None:
            for div in body.findall('.//tei:div', namespaces):
                # Get section heading
                head = div.find('tei:head', namespaces)
                section_title = head.text.strip() if head is not None and head.text else "Untitled Section"
                
                # Get section text
                section_text_parts = []
                for p in div.findall('.//tei:p', namespaces):
                    if p.text:
                        section_text_parts.append(p.text.strip())
                    # Also get text from nested elements
                    for elem in p.iter():
                        if elem.text and elem.tag not in ['{http://www.tei-c.org/ns/1.0}head']:
                            text = elem.text.strip()
                            if text and text not in section_text_parts:
                                section_text_parts.append(text)
                
                section_text = "\n".join(section_text_parts)
                if section_text.strip():
                    chunks.append((section_title, section_text))
        
        return {
            "title": title or "Untitled",
            "authors": authors_str,
            "abstract": abstract,
            "chunks": chunks
        }
        
    except ET.ParseError as e:
        logger.error(f"Failed to parse TEI XML: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing TEI XML: {e}")
        raise


def extract_text_from_grobid_result(grobid_result: dict) -> str:
    """
    Extract plain text from GROBID result dictionary.
    
    Args:
        grobid_result: Dictionary from parse_pdf_grobid()
    
    Returns:
        Plain text string with sections concatenated
    """
    chunks = grobid_result.get('chunks', [])
    if chunks:
        # Concatenate chunks with section headers
        chunk_texts = []
        for section_title, section_text in chunks:
            chunk_texts.append(f"{section_title}\n{section_text}")
        return "\n\n".join(chunk_texts)
    elif grobid_result.get('abstract'):
        return grobid_result['abstract']
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Parse PDF using GROBID and extract text to output file"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input PDF file"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output text file"
    )
    
    parser.add_argument(
        "--grobid-url",
        type=str,
        default="http://localhost:8070",
        help="GROBID server URL (default: http://localhost:8070)"
    )
    
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata (title, authors) at the top of output"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of plain text"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.suffix.lower() == '.pdf':
        logger.warning(f"Input file does not have .pdf extension: {args.input}")
    
    # Extract title from filename
    title = args.input.stem
    
    try:
        # Parse PDF with GROBID
        logger.info(f"Parsing PDF: {args.input}")
        grobid_result = parse_pdf_grobid(
            pdf_filepath=str(args.input),
            title=title,
            grobid_url=args.grobid_url
        )
        
        # Extract text
        if args.json:
            # Output as JSON
            output_content = json.dumps(grobid_result, indent=2, ensure_ascii=False)
        else:
            # Output as plain text
            text_parts = []
            
            if args.include_metadata:
                if grobid_result.get('title'):
                    text_parts.append(f"Title: {grobid_result['title']}")
                if grobid_result.get('authors'):
                    text_parts.append(f"Authors: {grobid_result['authors']}")
                if grobid_result.get('abstract'):
                    text_parts.append(f"\nAbstract:\n{grobid_result['abstract']}")
                if text_parts:
                    text_parts.append("\n" + "=" * 60 + "\n")
            
            # Add main content
            main_text = extract_text_from_grobid_result(grobid_result)
            text_parts.append(main_text)
            
            output_content = "\n".join(text_parts)
        
        # Write to output file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        # Log statistics
        char_count = len(output_content)
        chunk_count = len(grobid_result.get('chunks', []))
        logger.info(f"Successfully extracted {char_count:,} characters")
        logger.info(f"Output written to: {args.output}")
        if chunk_count > 0:
            logger.info(f"Extracted {chunk_count} sections")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
