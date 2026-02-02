#!/usr/bin/env python3
"""
Parser for "Talks with Sri Ramana Maharshi" - Messy Grobid Version
===================================================================

Handles the irregular formatting from grobid PDF extraction:
- Split markers: M.\n: or M.:\n
- Inline dialogues: D.: question M.: answer on same line
- Embedded questions: 'The man asked: "..."'
- Narrative contamination to filter out

Usage:
    python parse_talks_v2.py input.txt --output talks_sft.jsonl --stats
"""

import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # 'human' or 'assistant'
    content: str
    
    
@dataclass  
class Conversation:
    id: str
    turns: list[Turn] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Fix common grobid artifacts."""
    # Fix split markers: "M.\n:" -> "M.:"
    text = re.sub(r'M\.\s*\n\s*:', 'M.:', text)
    text = re.sub(r'D\.\s*\n\s*:', 'D.:', text)
    
    # Normalize various Maharshi markers to M.:
    text = re.sub(r'\bMaharshi\s*:', 'M.:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bBhagavan\s*:', 'M.:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSri Bhagavan\s*:', 'M.:', text, flags=re.IGNORECASE)
    
    # Normalize questioner markers to D.:
    text = re.sub(r'\bDevotee\s*:', 'D.:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bVisitor\s*:', 'D.:', text, flags=re.IGNORECASE)
    text = re.sub(r'\bQuestioner\s*:', 'D.:', text, flags=re.IGNORECASE)
    
    # Collapse multiple whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text


def is_narrative(text: str) -> bool:
    """Detect third-person narrative that shouldn't be dialogue."""
    narrative_patterns = [
        r'^He\s+(?:sought|found|was|had|did|went|came|left|sat|stood)',
        r'^She\s+(?:sought|found|was|had|did|went|came|left|sat|stood)',
        r'^The\s+(?:man|woman|visitor|devotee|questioner)\s+(?:was|had|did|went|came|left)',
        r'^This\s+(?:was|is|impressed|struck)',
        r'^Such\s+was',
        r'^Thus\s+',
        r'(?:acknowledged|impressed|confirmed|satisfied)\s+(?:the|his|her)',
        r'^There\s+(?:was|were|is|are)',
        r'^At\s+(?:length|last|once)',
        r'^When\s+Maharshi',
        r'^The\s+incident',
    ]
    
    text_stripped = text.strip()
    for pattern in narrative_patterns:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return True
    return False


def extract_embedded_question(text: str) -> tuple[str | None, str]:
    """
    Extract questions from narrative like:
    'The man asked softly: "It is said that..."'
    
    Returns (question, remaining_text) or (None, original_text)
    """
    patterns = [
        r'(?:asked|enquired|questioned|said)(?:\s+\w+)?:\s*["\']([^"\']+)["\']',
        r'(?:asked|enquired|questioned|said)(?:\s+\w+)?:\s*"([^"]+)"',
        r'(?:asked|enquired|questioned|said)(?:\s+\w+)?:\s*["\u201c]([^"\u201d]+)["\u201d]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            question = match.group(1).strip()
            remaining = text[match.end():].strip()
            return question, remaining
    
    return None, text


def split_inline_dialogue(text: str) -> list[tuple[str, str]]:
    """
    Split inline dialogues like:
    "D.: How to overcome? M.: By realising the Self. D.: But how?"
    
    Returns list of (marker, content) tuples.
    """
    # Pattern to match D.: or M.: markers
    pattern = r'(?:^|(?<=\s))([DM]\.:\s*)'
    
    parts = re.split(pattern, text)
    
    # parts will be: ['preamble', 'D.: ', 'content', 'M.: ', 'content', ...]
    results = []
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        
        if re.match(r'^[DM]\.:$', part.replace(' ', '')):
            # This is a marker
            marker = 'D' if part.startswith('D') else 'M'
            if i + 1 < len(parts):
                content = parts[i + 1].strip()
                if content:
                    results.append((marker, content))
            i += 2
        else:
            # Preamble or content without marker - check for embedded question
            if part:
                question, remaining = extract_embedded_question(part)
                if question:
                    results.append(('D', question))
            i += 1
    
    return results


def parse_talks(text: str) -> list[Conversation]:
    """Parse the full text into conversations."""
    
    # Normalize first
    text = normalize_text(text)
    
    # Split by Talk headers
    talk_pattern = r'Talk\s+(\d+)\.'
    
    sections = re.split(talk_pattern, text)
    # sections: [preamble, '1', content, '2', content, ...]
    
    conversations = []
    
    for i in range(1, len(sections), 2):
        talk_num = sections[i]
        if i + 1 >= len(sections):
            continue
        content = sections[i + 1]
        
        conv = parse_talk_section(content, talk_num)
        if conv and len(conv.turns) >= 2:
            conversations.append(conv)
    
    return conversations


def parse_talk_section(content: str, talk_num: str) -> Conversation | None:
    """Parse a single Talk section."""
    
    conv = Conversation(id=f"talk_{talk_num}")
    
    # Split into lines but also handle inline dialogues
    lines = content.split('\n')
    
    current_role = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip date headers
        if re.match(r'^\d+(?:st|nd|rd|th)\s+\w+,?\s+\d{4}', line):
            continue
        
        # Check for inline dialogues
        inline_parts = split_inline_dialogue(line)
        
        if inline_parts:
            # First, save any pending content
            if current_role and current_content:
                text = ' '.join(current_content).strip()
                if text and not is_narrative(text):
                    conv.turns.append(Turn(
                        role='human' if current_role == 'D' else 'assistant',
                        content=text
                    ))
                current_content = []
            
            # Process inline parts
            for marker, part_content in inline_parts:
                if is_narrative(part_content):
                    continue
                    
                role = 'human' if marker == 'D' else 'assistant'
                
                # If same role as last turn, append
                if conv.turns and conv.turns[-1].role == role:
                    conv.turns[-1].content += ' ' + part_content
                else:
                    conv.turns.append(Turn(role=role, content=part_content))
                
                current_role = marker
        else:
            # No markers found - might be continuation or narrative
            # Check for embedded questions in narrative
            question, _ = extract_embedded_question(line)
            if question:
                if current_role and current_content:
                    text = ' '.join(current_content).strip()
                    if text and not is_narrative(text):
                        conv.turns.append(Turn(
                            role='human' if current_role == 'D' else 'assistant',
                            content=text
                        ))
                    current_content = []
                
                conv.turns.append(Turn(role='human', content=question))
                current_role = 'D'
            elif current_role and not is_narrative(line):
                # Continuation of current speaker
                current_content.append(line)
    
    # Save any remaining content
    if current_role and current_content:
        text = ' '.join(current_content).strip()
        if text and not is_narrative(text):
            conv.turns.append(Turn(
                role='human' if current_role == 'D' else 'assistant',
                content=text
            ))
    
    # Clean up conversation
    conv = clean_conversation(conv)
    
    return conv

def clean_turn_content(text: str) -> str:
    """Remove trailing garbage from turn content."""
    # Remove trailing D.: or M.: markers (incomplete next turn)
    text = re.sub(r'\s*[DM]\.:\s*$', '', text)
    
    # Remove trailing "D.: [short word] D.:" patterns
    text = re.sub(r'\s*D\.:\s*\w{0,10}\s*D\.:\s*$', '', text)
    
    # Remove date headers
    text = re.sub(r'Talks with Sri Ramana Maharshi\s*\d+.*?\d{4}', '', text)
    text = re.sub(r'\d+(?:st|nd|rd|th)\s+\w+,?\s*\d{4}', '', text)
    
    # Remove isolated "D.:" or "M.:" mid-text followed by single word
    text = re.sub(r'\s+[DM]\.:\s+[DM]\.:', ' ', text)
    
    return text.strip()

def clean_conversation(conv: Conversation) -> Conversation:
    """Clean up a conversation: merge consecutive same-speaker, fix alternation."""
    
    if not conv.turns:
        return conv
    
    # Merge consecutive same-speaker turns
    merged = []
    for turn in conv.turns:
        if merged and merged[-1].role == turn.role:
            merged[-1].content += ' ' + turn.content
        else:
            merged.append(Turn(role=turn.role, content=turn.content))
    
    # Ensure starts with human
    while merged and merged[0].role != 'human':
        merged.pop(0)
    
    # Ensure ends with assistant
    while merged and merged[-1].role == 'human':
        merged.pop()
    
    # Clean up content
    for turn in merged:
        turn.content = clean_turn_content(turn.content)
        # Remove stray colons at start
        turn.content = re.sub(r'^:\s*', '', turn.content)
        # Collapse whitespace
        turn.content = ' '.join(turn.content.split())
    
    # Filter empty turns
    merged = [t for t in merged if len(clean_turn_content(t.content)) > 15]
    
    conv.turns = merged
    return conv


def to_messages_format(conv: Conversation) -> dict:
    """Convert to standard messages format."""
    return {
        'id': conv.id,
        'messages': [
            {'role': t.role, 'content': t.content}
            for t in conv.turns
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse 'Talks with Sri Ramana Maharshi' (grobid output)"
    )
    parser.add_argument('input', type=Path, help='Input text file')
    parser.add_argument('--output', '-o', type=Path, default=Path('talks_sft.jsonl'))
    parser.add_argument('--min-response', type=int, default=20,
                       help='Minimum chars in assistant response')
    parser.add_argument('--max-turns', type=int, default=12,
                       help='Max turns per conversation')
    parser.add_argument('--stats', action='store_true')
    
    args = parser.parse_args()
    
    text = args.input.read_text(encoding='utf-8', errors='replace')
    
    conversations = parse_talks(text)
    
    # Filter by response length
    filtered = []
    for conv in conversations:
        assistant_turns = [t for t in conv.turns if t.role == 'assistant']
        if assistant_turns and len(assistant_turns[0].content) >= args.min_response:
            filtered.append(conv)
    
    # Split long conversations
    final = []
    for conv in filtered:
        if len(conv.turns) <= args.max_turns:
            final.append(conv)
        else:
            # Split into chunks
            for i in range(0, len(conv.turns), args.max_turns - 2):
                chunk_turns = conv.turns[i:i + args.max_turns]
                # Ensure ends with assistant
                while chunk_turns and chunk_turns[-1].role == 'human':
                    chunk_turns.pop()
                if len(chunk_turns) >= 2:
                    final.append(Conversation(
                        id=f"{conv.id}_chunk{len(final)}",
                        turns=chunk_turns
                    ))
    
    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        for conv in final:
            f.write(json.dumps(to_messages_format(conv), ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(final)} conversations to {args.output}")
    
    if args.stats:
        total_turns = sum(len(c.turns) for c in final)
        human_chars = sum(len(t.content) for c in final for t in c.turns if t.role == 'human')
        assistant_chars = sum(len(t.content) for c in final for t in c.turns if t.role == 'assistant')
        
        print(f"\nStatistics:")
        print(f"  Conversations: {len(final)}")
        print(f"  Total turns: {total_turns}")
        print(f"  Avg turns/conv: {total_turns / len(final):.1f}")
        print(f"  Human chars: {human_chars:,}")
        print(f"  Assistant chars: {assistant_chars:,}")
        print(f"  Estimated tokens: {(human_chars + assistant_chars) // 4:,}")
        
        print(f"\nSample conversations:")
        for conv in final[:3]:
            print(f"\n  [{conv.id}]")
            for turn in conv.turns[:4]:
                preview = turn.content[:70] + ('...' if len(turn.content) > 70 else '')
                print(f"    {turn.role}: {preview}")


if __name__ == '__main__':
    main()