#!/usr/bin/env python3
"""
Ramana Maharshi Q&A Extraction Pipeline

Cleans OCR-extracted text from Paul Brunton's "Commentaries by Maharishee"
and extracts interchange pairs suitable for fine-tuning.

Outputs:
  - JSONL for training (one {"question":..., "answer":...} per line)
  - JSON for review (formatted, with metadata)
  - Summary stats

Usage:
    python ramana_extract.py input.pdf
    python ramana_extract.py input.pdf --format jsonl -o training.jsonl
    python ramana_extract.py input.pdf --format json -o compact.json
    python ramana_extract.py input.pdf --include-teachings
"""

import re
import json
import subprocess
import argparse
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Use pdftotext to extract with layout preservation."""
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout


# ---------------------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Apply successive cleaning passes to raw OCR text."""
    text = raw

    # Remove form feed characters
    text = text.replace('\f', '\n')

    # Remove page continuation markers
    text = re.sub(
        r'\(continued from the previous page\)\s*',
        '', text, flags=re.IGNORECASE
    )

    # Remove page headers: right-aligned chapter titles and page numbers
    text = re.sub(r'^\s{20,}\d+\s*$', '', text, flags=re.MULTILINE)

    # All known chapter headings (right-aligned in the PDF layout)
    chapter_headings = [
        "BEYOND YOGA", "FALLACIES OF RELIGION", "THE MEANING OF RELIGION",
        "THE MEANING OF MYSTICISM", "THE MEANING OF PHILOSOPHY",
        "CHARACTERISTICS OF PHILOSOPHIC DISCIPLINE", "UNLISTED",
        "PHYSIOLOGY OF SENSATION AND PERCEPTION",
        "ILLUSIONS OF SPACE TIME AND EXTERNALITY",
        "DOCTRINE OF MENTALISM", "THE ILLUSION OF WORLD EXPERIENCE",
        "THE ILLUSION OF EGO EXPERIENCE", "AVASTATRAYA",
        "THE ULTIMATE AS TRUTH", "PRACTICAL PHILOSOPHY",
        "SAGEHOOD AS AN IDEAL", "DOCTRINE OF NON-CAUSALITY",
        "THE MIND", "THE ULTIMATE AS REALITY",
        "EASTERN AND WESTERN THINKERS",
        "EASTERN AND WESTERN SCHOOLS OF THOUGHT",
        "THE NEED OF ULTRA-MYSTICISM", "CONTENTS",
    ]
    heading_pat = '|'.join(re.escape(h) for h in chapter_headings)
    text = re.sub(
        rf'^\s{{20,}}({heading_pat}).*$',
        '', text, flags=re.MULTILINE
    )

    # Remove bracketed page/editor markers like [1], [7], [9]14, [207]221
    text = re.sub(r'^\s*\[\d+\]\d*\s*$', '', text, flags=re.MULTILINE)

    # Remove footnotes at bottom of pages (can span multiple lines)
    # First pass: numbered footnotes with known patterns
    text = re.sub(
        r'^\d{1,3}\s+(The original editor|Blank page|Back cover|Front cover|'
        r'Void page|Picture of|The paras in|"of,"|"[Gg]nana"|'
        r'This para|PB himself|PB inserted|Handwritten|'
        r'A blank page|These paras|The word|We have|Incomplete|'
        r'This word|Typed above|PB changed|Para \d).*$',
        '', text, flags=re.MULTILINE
    )
    # Second pass: multi-line footnotes (numbered line followed by continuation
    # lines that are clearly editorial, e.g. "vision, M said: ...")
    text = re.sub(
        r'^\d{1,3}\s+The original editor inserted\b.*?(?=\n\n|\n\(\d)',
        '', text, flags=re.MULTILINE | re.DOTALL
    )

    # Remove "Chapter N: Title" lines
    text = re.sub(r'^\s*Chapter \d+:.*$', '', text, flags=re.MULTILINE)

    # Normalize Maharshi spelling variants
    text = text.replace('Maharishee', 'Maharshi')
    text = text.replace('Maharishi', 'Maharshi')

    # Fix common OCR / typographic artifacts
    text = text.replace(' – ', '—')
    text = text.replace('–', '—')

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix broken words across lines (hyphenated line breaks before lowercase)
    text = re.sub(r'-\s*\n\s*(?=[a-z])', '', text)

    return text


# ---------------------------------------------------------------------------
# 3. PARAGRAPH EXTRACTION
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    """A numbered paragraph from the text."""
    ref: str          # e.g. "12-1"
    page: int         # page number from ref
    para: int         # para number from ref
    raw_text: str     # cleaned text


def extract_paragraphs(text: str) -> list[Paragraph]:
    """Split text into numbered paragraphs using (page-para) markers."""
    pattern = r'\((\d{1,3})-(\d{1,2})\)'
    matches = list(re.finditer(pattern, text))

    paragraphs = []
    for i, match in enumerate(matches):
        page = int(match.group(1))
        para = int(match.group(2))
        ref = f"{page}-{para}"

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        raw = text[start:end].strip()
        raw = clean_paragraph(raw)

        if raw and len(raw) > 10:
            paragraphs.append(Paragraph(
                ref=ref, page=page, para=para, raw_text=raw
            ))

    return paragraphs


def clean_paragraph(text: str) -> str:
    """Clean an individual paragraph's text."""
    # Remove stray page numbers on their own line
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Remove trailing footnote numbers stuck to words
    text = re.sub(r'(?<=[\w.?!"])\d{1,3}(?=\s*$)', '', text, flags=re.MULTILINE)

    # Unwrap lines intelligently
    lines = text.split('\n')
    joined = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if joined and joined[-1] != '':
                joined.append('')
            continue

        # Keep Q: and A: lines separate
        if re.match(r'^[QA]:', stripped):
            joined.append(stripped)
        # If previous line exists and this continues a sentence
        elif (joined and joined[-1]
              and not joined[-1].endswith(('.', '?', '!', '"', ':', '"'))
              and len(stripped) > 0
              and (stripped[0].islower() or stripped[0] in '("\'—')):
            joined[-1] = joined[-1].rstrip() + ' ' + stripped
        # Indented continuation (common for A: responses)
        elif (joined and joined[-1]
              and len(line) > 0 and line[0] == ' '
              and not re.match(r'^\s*[QA]:', line)):
            joined[-1] = joined[-1].rstrip() + ' ' + stripped
        else:
            joined.append(stripped)

    text = '\n'.join(joined)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# 4. INTERCHANGE EXTRACTION
# ---------------------------------------------------------------------------

@dataclass
class Interchange:
    """A Q&A exchange with Ramana Maharshi."""
    ref: str
    question: str
    answer: str
    questioner: str = ""
    topic_tags: list[str] = field(default_factory=list)
    interchange_type: str = "direct_qa"
    turns: list[dict] = field(default_factory=list)


@dataclass
class Teaching:
    """A non-Q&A teaching statement."""
    ref: str
    text: str
    topic_tags: list[str] = field(default_factory=list)


def extract_interchanges(paragraphs: list[Paragraph]) -> tuple[list[Interchange], list[Teaching]]:
    """Extract Q&A interchanges and standalone teachings."""
    interchanges = []
    teachings = []

    for para in paragraphs:
        text = para.raw_text

        # --- Pattern 1: Direct Q: / A: exchanges ---
        if 'Q:' in text and 'A:' in text:
            pairs = extract_qa_pairs(text)
            if pairs:
                if len(pairs) == 1:
                    q, a = pairs[0]
                    interchanges.append(Interchange(
                        ref=para.ref,
                        question=q.strip(),
                        answer=a.strip(),
                        topic_tags=auto_tag(q + ' ' + a),
                        interchange_type="direct_qa"
                    ))
                else:
                    turns = []
                    for q, a in pairs:
                        if q.strip():
                            turns.append({"role": "questioner", "text": q.strip()})
                        if a.strip():
                            turns.append({"role": "ramana", "text": a.strip()})
                    interchanges.append(Interchange(
                        ref=para.ref,
                        question=pairs[0][0].strip(),
                        answer=pairs[0][1].strip(),
                        topic_tags=auto_tag(text),
                        interchange_type="multi_turn",
                        turns=turns
                    ))

        # --- Pattern 2: Reported exchanges: "M said", "Maharshi replied" ---
        elif re.search(r'\bM said\b|Maharshi\s+(?:said|replied|told|answered)', text):
            q, a = extract_reported_exchange(text)
            # Validate: the answer should not be a near-duplicate of another's answer
            # and the "M said" should appear in the first half of the paragraph
            m_pos = re.search(r'\bM said\b|Maharshi\s+(?:said|replied|told|answered)', text)
            if q and a and m_pos and m_pos.start() < len(text) * 0.7:
                interchanges.append(Interchange(
                    ref=para.ref,
                    question=q.strip(),
                    answer=a.strip(),
                    topic_tags=auto_tag(text),
                    interchange_type="reported_exchange"
                ))
            elif is_teaching(text):
                teachings.append(Teaching(
                    ref=para.ref, text=text, topic_tags=auto_tag(text)
                ))

        # --- Pattern 3: Visitor/devotee exchange ---
        elif re.search(r'(?:A visitor|A devotee|Someone|One person|A disciple)\s+(?:asked|said|told)', text, re.I):
            q, a = extract_visitor_exchange(text)
            if q and a:
                interchanges.append(Interchange(
                    ref=para.ref,
                    question=q.strip(),
                    answer=a.strip(),
                    topic_tags=auto_tag(text),
                    interchange_type="reported_exchange"
                ))

        # --- Standalone teaching ---
        else:
            if is_teaching(text):
                teachings.append(Teaching(
                    ref=para.ref,
                    text=text,
                    topic_tags=auto_tag(text)
                ))

    return interchanges, teachings


def extract_qa_pairs(text: str) -> list[tuple[str, str]]:
    """Extract one or more Q:/A: pairs from text."""
    # Split on Q: and A: markers
    parts = re.split(r'(?:^|\n)\s*([QA]):\s*', text)

    pairs = []
    current_q = None
    preamble = parts[0].strip() if parts else ""

    i = 1
    while i < len(parts) - 1:
        marker = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if marker == 'Q':
            current_q = content
        elif marker == 'A':
            if current_q is not None:
                pairs.append((current_q, content))
                current_q = None
            elif preamble:
                pairs.append((preamble, content))
        i += 2

    return pairs


def extract_reported_exchange(text: str) -> tuple[str, str]:
    """Extract Q&A from 'M said: ...' patterns."""
    match = re.search(
        r'(.*?)(?:M said|Maharshi\s+(?:said|replied|told|answered))[:\s]*["""]?(.*)',
        text, re.DOTALL
    )
    if match:
        context = match.group(1).strip()
        response = match.group(2).strip().strip('""\'"')
        return context, response
    return "", ""


def extract_visitor_exchange(text: str) -> tuple[str, str]:
    """Extract from 'A visitor asked...' patterns."""
    match = re.search(
        r'((?:A visitor|A devotee|Someone|One person|A disciple)\s+(?:asked|said|told).*?)'
        r'(?:Maharshi|M|Sri Bhagavan|He)\s+(?:said|replied|answered|told)[:\s]*(.*)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


# ---------------------------------------------------------------------------
# 5. AUTO-TAGGING
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS = {
    "self_inquiry": ["Self", "enquiry", "enquire", "vichara", "who am I", "I-thought"],
    "meditation": ["meditation", "dhyana", "meditate", "concentrate", "concentration"],
    "mind": ["mind", "thought", "thoughts", "mental", "vasanas", "samskaras"],
    "samadhi": ["samadhi", "nirvikalpa", "savikalpa", "sahaja", "trance"],
    "devotion": ["devotion", "bhakti", "worship", "God", "Iswara", "surrender"],
    "visions": ["vision", "visions", "psychic", "siddhis", "occult", "clairvoyance"],
    "silence": ["silence", "mouna", "stillness", "peace"],
    "guru": ["guru", "grace", "Bhagavan", "master", "teacher"],
    "world_illusion": ["maya", "illusion", "dream", "unreality", "unreal", "phenomenal"],
    "heart": ["Heart", "heart", "hridayam"],
    "kundalini": ["kundalini", "chakra", "chakras", "nadi", "nadis"],
    "scripture": ["scripture", "Vedas", "Gita", "Upanishad"],
    "jnana": ["jnana", "knowledge", "gnana", "wisdom"],
    "ego": ["ego", "I-thought", "ahankara", "ahamkara"],
    "death": ["death", "dying", "dead", "rebirth", "reincarnation"],
    "sleep": ["sleep", "waking", "dream", "deep sleep", "sushupti"],
    "practice": ["practice", "sadhana", "effort", "method"],
    "food_conduct": ["food", "vegetarian", "meat", "diet", "smoking", "alcohol"],
}


def auto_tag(text: str) -> list[str]:
    """Assign topic tags based on keyword presence."""
    tags = []
    text_lower = text.lower()
    for tag, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                tags.append(tag)
                break
    return sorted(set(tags))


def is_teaching(text: str) -> bool:
    """Heuristic: is this a teaching vs editorial/metadata?"""
    if len(text) < 40:
        return False
    if re.match(r'^(Editor|Note|See also|Cf\.|The original|Contents)', text, re.I):
        return False
    if re.match(r'^Chapter\s+\d', text):
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False
    return True


# ---------------------------------------------------------------------------
# 6. POST-PROCESSING
# ---------------------------------------------------------------------------

def postprocess_interchanges(interchanges: list[Interchange]) -> list[Interchange]:
    """Clean up remaining artifacts in extracted interchanges."""
    cleaned = []
    for ic in interchanges:
        # Clean bracket artifacts from questions
        ic.question = re.sub(r'^\[+\s*', '', ic.question)
        ic.question = re.sub(r'\s*\]+$', '', ic.question)
        # Remove stray Q: at start of question (from bracket patterns like [Q: ...])
        ic.question = re.sub(r'^Q:\s*', '', ic.question)

        # Clean footnote number artifacts (e.g. trailing ]148 or )148)
        ic.question = re.sub(r'\]\d{1,3}\s*', '', ic.question)
        ic.answer = re.sub(r'\]\d{1,3}\s*', '', ic.answer)

        # Remove footnote leaks in answers
        ic.answer = re.sub(r'\s*at the bottom of the page.*$', '', ic.answer)
        ic.answer = re.sub(r'\s*The original editor.*$', '', ic.answer)
        ic.answer = re.sub(r'\s*\d+\s+The original editor.*$', '', ic.answer)

        # Clean trailing bracket from bracket-enclosed exchanges
        ic.answer = re.sub(r'\s*\"\]?\s*$', '"', ic.answer)
        ic.answer = re.sub(r'\]\s*$', '', ic.answer)

        # Clean multi-turn exchanges too
        for turn in ic.turns:
            turn["text"] = re.sub(r'^\[+\s*', '', turn["text"])
            turn["text"] = re.sub(r'\s*\]+$', '', turn["text"])
            turn["text"] = re.sub(r'^Q:\s*', '', turn["text"])
            turn["text"] = re.sub(r'\]\d{1,3}\s*', '', turn["text"])

        # Strip whitespace
        ic.question = ic.question.strip()
        ic.answer = ic.answer.strip()

        # Skip if either side is now empty
        if ic.question and ic.answer:
            cleaned.append(ic)

    return cleaned


# ---------------------------------------------------------------------------
# 7. OUTPUT
# ---------------------------------------------------------------------------

def format_for_training(interchanges: list[Interchange]) -> list[dict]:
    """Format as training examples."""
    examples = []
    for ic in interchanges:
        if ic.interchange_type == "multi_turn" and ic.turns:
            examples.append({
                "ref": ic.ref,
                "type": "multi_turn",
                "turns": ic.turns,
                "tags": ic.topic_tags
            })
        else:
            examples.append({
                "ref": ic.ref,
                "type": ic.interchange_type,
                "question": ic.question,
                "answer": ic.answer,
                "tags": ic.topic_tags
            })
    return examples


def format_for_review(interchanges: list[Interchange],
                      teachings: list[Teaching]) -> dict:
    """Format with full metadata for human review."""
    return {
        "metadata": {
            "source": "Commentaries by Sri Ramana Maharshi",
            "recorded_by": "Paul Brunton & Munagala S. Venkataramiah",
            "total_interchanges": len(interchanges),
            "total_teachings": len(teachings),
            "type_breakdown": {
                "direct_qa": sum(1 for i in interchanges if i.interchange_type == "direct_qa"),
                "multi_turn": sum(1 for i in interchanges if i.interchange_type == "multi_turn"),
                "reported": sum(1 for i in interchanges if i.interchange_type == "reported_exchange"),
            }
        },
        "interchanges": [asdict(ic) for ic in interchanges],
        "teachings": [asdict(t) for t in teachings]
    }


def print_summary(interchanges: list[Interchange], teachings: list[Teaching]):
    """Print extraction summary."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Ramana Maharshi Q&A Extraction Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Total interchanges:  {len(interchanges)}", file=sys.stderr)
    direct = sum(1 for i in interchanges if i.interchange_type == 'direct_qa')
    multi = sum(1 for i in interchanges if i.interchange_type == 'multi_turn')
    reported = sum(1 for i in interchanges if i.interchange_type == 'reported_exchange')
    print(f"    Direct Q&A:        {direct}", file=sys.stderr)
    print(f"    Multi-turn:        {multi}", file=sys.stderr)
    print(f"    Reported exchange: {reported}", file=sys.stderr)
    print(f"  Standalone teachings: {len(teachings)}", file=sys.stderr)

    all_tags = {}
    for ic in interchanges:
        for tag in ic.topic_tags:
            all_tags[tag] = all_tags.get(tag, 0) + 1
    if all_tags:
        print(f"\n  Topic distribution:", file=sys.stderr)
        for tag, count in sorted(all_tags.items(), key=lambda x: -x[1])[:15]:
            bar = '█' * (count // 2)
            print(f"    {tag:25s} {count:3d} {bar}", file=sys.stderr)

    if interchanges:
        print(f"\n  --- Sample interchange ({interchanges[0].ref}) ---", file=sys.stderr)
        print(f"  Q: {interchanges[0].question[:120]}", file=sys.stderr)
        print(f"  A: {interchanges[0].answer[:120]}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Q&A interchanges from Ramana Maharshi Commentaries"
    )
    parser.add_argument("input", help="Path to PDF or pre-extracted .txt file")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["jsonl", "json", "review"],
                        default="review",
                        help="Output format (default: review)")
    parser.add_argument("--include-teachings", action="store_true",
                        help="Include standalone teachings in output")
    parser.add_argument("--min-answer-length", type=int, default=20,
                        help="Minimum answer length in chars (default: 20)")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Extract or load text
    if input_path.suffix.lower() == '.pdf':
        print(f"Extracting text from {input_path}...", file=sys.stderr)
        raw_text = extract_text_from_pdf(str(input_path))
    else:
        raw_text = input_path.read_text()

    # Clean
    print("Cleaning text...", file=sys.stderr)
    cleaned = clean_text(raw_text)

    # Extract paragraphs
    print("Extracting paragraphs...", file=sys.stderr)
    paragraphs = extract_paragraphs(cleaned)
    print(f"  Found {len(paragraphs)} paragraphs", file=sys.stderr)

    # Extract interchanges
    print("Extracting interchanges...", file=sys.stderr)
    interchanges, teachings = extract_interchanges(paragraphs)

    # Post-process: clean remaining artifacts
    interchanges = postprocess_interchanges(interchanges)

    # Filter by minimum answer length
    interchanges = [ic for ic in interchanges
                    if len(ic.answer) >= args.min_answer_length]

    print_summary(interchanges, teachings)

    # Output
    out_file = open(args.output, 'w') if args.output else sys.stdout

    if args.format == "jsonl":
        examples = format_for_training(interchanges)
        if args.include_teachings:
            for t in teachings:
                examples.append({
                    "ref": t.ref, "type": "teaching",
                    "text": t.text, "tags": t.topic_tags
                })
        for ex in examples:
            print(json.dumps(ex, ensure_ascii=False), file=out_file)

    elif args.format == "json":
        examples = format_for_training(interchanges)
        print(json.dumps(examples, indent=2, ensure_ascii=False), file=out_file)

    elif args.format == "review":
        review = format_for_review(interchanges, teachings)
        print(json.dumps(review, indent=2, ensure_ascii=False), file=out_file)

    if args.output:
        out_file.close()
        print(f"Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
