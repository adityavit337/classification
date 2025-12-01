"""Text processing for Qwen3-VL OCR output to DeBERTa classification format.

Provides intelligent page break handling to merge questions/answers split across pages.
"""

import re
import json
import csv
import unicodedata
from typing import List, Tuple, Dict, Optional
from difflib import SequenceMatcher
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Process Qwen3-VL OCR output for DeBERTa classification.
    
    Features:
    - Intelligent page break handling (merges questions/answers split across pages)
    - OCR artifact correction for handwritten text
    - Clean line-by-line format matching DeBERTa training data structure
    """
    
    def __init__(self, similarity_threshold: float = 0.8, 
                 remove_bounding_boxes: bool = True,
                 aggressive_ocr_correction: bool = False):
        """
        Initialize text processor.
        
        Args:
            similarity_threshold: Threshold for determining if pages should be merged (0-1)
            remove_bounding_boxes: Remove DeepSeek bounding box annotations (default: True)
            aggressive_ocr_correction: Apply aggressive OCR corrections for handwritten text
        """
        self.similarity_threshold = similarity_threshold
        self.remove_bounding_boxes = remove_bounding_boxes
        self.aggressive_ocr_correction = aggressive_ocr_correction
        
        # Common OCR misrecognition patterns for handwritten text
        self.ocr_corrections = {
            r'\bl\b': 'I',  # lowercase L often mistaken for I
            r'0(?=[a-zA-Z])': 'O',  # zero before letters likely O
            r'(?<=[a-zA-Z])0': 'o',  # zero after letters likely o
            r'rn': 'm',  # 'rn' often mistaken for 'm'
            r'vv': 'w',  # 'vv' often mistaken for 'w'
            r'\|': 'I',  # pipe often mistaken for I
            r'`': "'",  # backtick to apostrophe
        }
        
        # Common word-level OCR errors
        self.word_corrections = {
            r'\btbe\b': 'the',
            r'\bteh\b': 'the',
            r'\bwiht\b': 'with',
            r'\bwhcih\b': 'which',
            r'\bthat\s+is\b': 'that is',
            r'\bi\s+am\b': 'I am',
            r'\bi\s+have\b': 'I have',
        }
        
        # Page break patterns common in OCR output
        self.page_break_patterns = [
            r'-{3,}',  # Multiple dashes
            r'_{3,}',  # Multiple underscores
            r'={3,}',  # Multiple equals
            r'\[?page\s*\d*\]?',  # [page 1], page 2, etc.
            r'---\s*page\s*break\s*---',
            r'\f',  # Form feed character
            r'\x0c',  # Form feed hex
        ]
        
    # =========================================================================
    # OCR TEXT PREPROCESSING METHODS
    # =========================================================================
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to ASCII equivalents where possible."""
        normalized = unicodedata.normalize('NFKD', text)
        
        # Common unicode replacements
        replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '–': '-', '—': '-',  # En/Em dashes
            '…': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
            '\u200b': '',  # Zero-width space
            '\u2028': '\n',  # Line separator
            '\u2029': '\n\n',  # Paragraph separator
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
            
        return normalized
    
    def fix_line_breaks(self, text: str) -> str:
        """
        Fix line breaks that occur mid-sentence in handwritten OCR.
        Preserves intentional paragraph breaks.
        """
        # Replace multiple newlines with placeholder
        text = re.sub(r'\n{2,}', '<<PARA>>', text)
        
        # Handle hyphenated line breaks (word split across lines)
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Replace single newlines with space (mid-sentence breaks)
        text = re.sub(r'\n', ' ', text)
        
        # Restore paragraph breaks
        text = text.replace('<<PARA>>', '\n\n')
        
        return text
    
    def fix_spacing(self, text: str) -> str:
        """Fix spacing issues common in handwritten OCR."""
        # Remove extra spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # No space before punctuation
        text = re.sub(r'([.,!?;:])(?=[A-Za-z])', r'\1 ', text)  # Space after punctuation
        
        # Fix spacing around quotes
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR misrecognition corrections."""
        for pattern, replacement in self.ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def apply_word_corrections(self, text: str) -> str:
        """Fix common word-level OCR errors in handwritten text."""
        for pattern, replacement in self.word_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def remove_noise(self, text: str) -> str:
        """Remove noise characters common in handwritten OCR."""
        # Remove isolated special characters
        text = re.sub(r'(?<!\w)[^\w\s.,!?;:\'"()-](?!\w)', '', text)
        
        # Remove repeated punctuation (except ellipsis)
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        
        # Remove stray marks that are likely OCR artifacts
        text = re.sub(r'\s[~`^*#@$%&]\s', ' ', text)
        
        return text
    
    def capitalize_sentences(self, text: str) -> str:
        """Ensure proper sentence capitalization."""
        if not text:
            return text
            
        # Capitalize first letter of text
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence-ending punctuation
        text = re.sub(
            r'([.!?]\s+)([a-z])',
            lambda m: m.group(1) + m.group(2).upper(),
            text
        )
        
        # Capitalize 'I' when standalone
        text = re.sub(r'\bi\b', 'I', text)
        
        return text
    
    def preprocess_ocr_text(self, text: str, aggressive: bool = None) -> str:
        """
        Main OCR preprocessing pipeline for handwritten text.
        
        Args:
            text: Raw OCR extracted text
            aggressive: If True, apply more aggressive corrections (overrides init setting)
            
        Returns:
            Cleaned and preprocessed text
        """
        if not text or not text.strip():
            return ""
        
        if aggressive is None:
            aggressive = self.aggressive_ocr_correction
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Remove page breaks (using patterns)
        for pattern in self.page_break_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Step 3: Fix line breaks
        text = self.fix_line_breaks(text)
        
        # Step 4: Remove noise
        text = self.remove_noise(text)
        
        # Step 5: Fix spacing
        text = self.fix_spacing(text)
        
        if aggressive:
            # Step 6: Apply OCR character corrections
            text = self.apply_ocr_corrections(text)
            
            # Step 7: Fix common words
            text = self.apply_word_corrections(text)
        
        # Step 8: Capitalize sentences
        text = self.capitalize_sentences(text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split preprocessed text into individual sentences for classification."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def extract_text_segments(self, text: str, min_length: int = 10) -> List[Dict]:
        """
        Extract text segments suitable for classification.
        Returns list of dicts with text and metadata.
        
        Args:
            text: Raw OCR text
            min_length: Minimum character length for segments
            
        Returns:
            List of segment dictionaries with id, text, counts, and likely_type
        """
        preprocessed = self.preprocess_ocr_text(text)
        sentences = self.split_into_sentences(preprocessed)
        
        segments = []
        for idx, sentence in enumerate(sentences):
            if len(sentence) >= min_length:
                # Detect if likely question or statement based on punctuation
                ends_with_question = sentence.rstrip().endswith('?')
                
                segments.append({
                    'id': idx + 1,
                    'text': sentence,
                    'char_count': len(sentence),
                    'word_count': len(sentence.split()),
                    'likely_type': 'question' if ends_with_question else 'statement'
                })
        
        return segments
    
    def preprocess_with_page_awareness(self, text: str, aggressive: bool = True) -> str:
        """
        Preprocess OCR text with intelligent page break handling.
        
        Handles questions/answers split across pages by:
        1. Detecting incomplete sentences at page boundaries
        2. Merging continuation text from the next page
        3. Preserving Q/A boundaries (never merging Q into A or vice versa)
        
        Args:
            text: Raw OCR text with page markers (--- PAGE X ---)
            aggressive: Apply aggressive OCR corrections
            
        Returns:
            Preprocessed text with intelligently merged page content
        """
        # Pattern used by OCRExtractor
        page_pattern = r'---\s*PAGE\s*(\d+)\s*---'
        
        # Split by page separators, keeping track of page numbers
        parts = re.split(page_pattern, text)
        
        # Extract page contents (parts alternates: content, page_num, content, page_num, ...)
        page_contents = []
        i = 0
        while i < len(parts):
            content = parts[i].strip()
            if content:
                page_contents.append(content)
            i += 1
            # Skip the page number capture group
            if i < len(parts) and parts[i].isdigit():
                i += 1
        
        if not page_contents:
            return self.preprocess_ocr_text(text, aggressive=aggressive)
        
        # Intelligently merge pages where content is split
        merged_pages = self._intelligent_page_merge(page_contents)
        
        # Apply preprocessing to each merged page
        processed_pages = []
        for content in merged_pages:
            processed_content = self.preprocess_ocr_text(content, aggressive=aggressive)
            if processed_content:
                processed_pages.append(processed_content)
        
        # Join pages with paragraph breaks
        return '\n\n'.join(processed_pages)
    
    def _intelligent_page_merge(self, page_contents: List[str]) -> List[str]:
        """
        Intelligently merge page contents where questions/answers are split.
        
        Strategy:
        1. Check if page ends with incomplete sentence
        2. Check if next page starts with continuation (lowercase, no Q/A marker)
        3. Merge if both conditions met AND types are compatible
        
        Args:
            page_contents: List of text content from each page
            
        Returns:
            List of merged page contents
        """
        if len(page_contents) <= 1:
            return page_contents
        
        merged = []
        i = 0
        
        while i < len(page_contents):
            current_page = page_contents[i]
            
            # Check if we should merge with next page(s)
            while i + 1 < len(page_contents):
                next_page = page_contents[i + 1]
                
                if self._should_merge_across_pages(current_page, next_page):
                    # Merge: remove trailing hyphen if present, add space
                    if current_page.rstrip().endswith('-'):
                        current_page = current_page.rstrip()[:-1] + next_page.lstrip()
                    else:
                        current_page = current_page.rstrip() + ' ' + next_page.lstrip()
                    
                    logger.info(f"Merged page {i+1} with page {i+2} (split content detected)")
                    i += 1
                else:
                    break
            
            merged.append(current_page)
            i += 1
        
        logger.info(f"Page merge: {len(page_contents)} pages -> {len(merged)} merged sections")
        return merged
    
    def _should_merge_across_pages(self, page1: str, page2: str) -> bool:
        """
        Determine if two consecutive pages should be merged.
        
        Conditions for merging:
        1. Page 1 ends with incomplete sentence
        2. Page 2 starts with continuation (not a new Q/A)
        3. Types are compatible (not switching from Q to A or vice versa)
        
        Args:
            page1: Content of first page
            page2: Content of second page
            
        Returns:
            True if pages should be merged
        """
        if not page1 or not page2:
            return False
        
        # Get last line of page 1 and first line of page 2
        page1_lines = [l.strip() for l in page1.split('\n') if l.strip()]
        page2_lines = [l.strip() for l in page2.split('\n') if l.strip()]
        
        if not page1_lines or not page2_lines:
            return False
        
        last_line = page1_lines[-1]
        first_line = page2_lines[0]
        
        # Condition 1: Check if page 1 ends incompletely
        if not self.is_incomplete_sentence(last_line):
            return False
        
        # Condition 2: Check if page 2 starts with a NEW Q/A marker
        new_qa_pattern = r'^(q[\s\.\(\d]|ques[\s\.]|question[\s\.]|ans[\s\.\:]|answer[\s\.\:]|a[\s\.\:]|\d+[\s\.\)]+[qQaA])'
        if re.match(new_qa_pattern, first_line, re.IGNORECASE):
            # Next page starts a new question/answer - don't merge
            return False
        
        # Condition 3: Check type compatibility
        last_type = self.get_line_type(last_line)
        first_type = self.get_line_type(first_line)
        
        # Don't merge if switching between question and answer
        if last_type == 'question' and first_type == 'answer':
            return False
        if last_type == 'answer' and first_type == 'question':
            return False
        
        # Additional check: if next page starts with lowercase or continuation words
        continuation_indicators = [
            first_line[0].islower() if first_line else False,
            first_line.startswith(('and ', 'or ', 'but ', 'because ', 'which ', 'that ', 'the ', 'a ', 'an ')),
            first_line.startswith(('is ', 'are ', 'was ', 'were ', 'has ', 'have ', 'had ')),
        ]
        
        # Merge if any continuation indicator is present
        return any(continuation_indicators)

    # =========================================================================
    # PAGE MARKER AND ANNOTATION DETECTION
    # =========================================================================
        
    def is_page_marker(self, text: str) -> bool:
        """
        Detect if a line is a page break marker.
        
        Common patterns:
        - PAGE 1, Page 2, page 3
        - Page numbers alone: "1", "12", "145"
        - Date stamps: "12/11/2024", "2025-11-21"
        - Separators: "---", "===", "......"
        - Metadata: "Total Pages: 8", "Extracted on: ..."
        - Section headers: "Chapter 1", "Unit 2"
        
        Args:
            text: Line to check
            
        Returns:
            True if line is a page marker/separator
        """
        text_lower = text.lower().strip()
        
        # Pattern 1: Explicit page markers
        if re.match(r'^page\s*\d+', text_lower):
            return True
        
        # Pattern 2: Standalone numbers (likely page numbers)
        if re.match(r'^\d+$', text.strip()) and len(text.strip()) <= 4:
            return True
        
        # Pattern 3: Date patterns
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
            return True
        if re.match(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', text):
            return True
        
        # Pattern 4: Separators (repeated characters)
        if re.match(r'^[-=_.]{3,}$', text.strip()):
            return True
        
        # Pattern 5: Common OCR metadata
        metadata_patterns = [
            'total pages', 'extracted on', 'extracted text',
            'pdf:', 'fig', 'figure', 'image of', 'diagram',
            'not to be republished', 'rationalised', '© ncert'
        ]
        if any(pattern in text_lower for pattern in metadata_patterns):
            return True
        
        # Pattern 6: Section headers
        if re.match(r'^(chapter|unit|section|part|appendix)\s*\d*[:\s]', text_lower):
            return True
        
        return False
    
    def remove_deepseek_annotations(self, text: str) -> str:
        """
        Remove DeepSeek OCR bounding box annotations while preserving content.
        
        Removes patterns like:
        - <|ref|>text<|/ref|><|det|>[[120, 163, 603, 197]]<|/det|>
        - <|ref|>image<|/ref|><|det|>[[280, 50, 721, 160]]<|/det|>
        
        Args:
            text: Raw DeepSeek markdown with annotations
            
        Returns:
            Clean text without bounding box annotations
        """
        if not text:
            return ""
        
        # Remove bounding box references: <|ref|>...<|/ref|><|det|>[[...]]<|/det|>
        text = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>\s*', '', text)
        
        # Remove any remaining annotation tags
        text = re.sub(r'<\|[^|]+\|>', '', text)
        
        return text
    
    def is_incomplete_sentence(self, text: str) -> bool:
        """
        Check if a sentence is incomplete (likely cut off by page break).
        
        Indicators:
        - Ends without punctuation
        - Ends with conjunctions (and, or, but)
        - Ends with prepositions (of, in, to, from)
        - Ends with articles (a, an, the)
        - Ends with comma
        
        Args:
            text: Text line to check
            
        Returns:
            True if sentence appears incomplete
        """
        text = text.strip()
        
        if not text:
            return False
        
        # If it ends with proper punctuation, likely complete
        if text[-1] in '.!?":)]':
            return False
        
        # Check last word
        words = text.split()
        if not words:
            return False
        
        last_word = words[-1].lower().rstrip('.,;:')
        
        # Incomplete indicators
        incomplete_endings = [
            # Articles
            'a', 'an', 'the',
            # Prepositions
            'of', 'in', 'to', 'from', 'with', 'at', 'by', 'for', 'on',
            # Conjunctions
            'and', 'or', 'but', 'because', 'as', 'since', 'when', 'if',
            # Auxiliary verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'do', 'does', 'did',
            'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could',
            # Pronouns that suggest continuation
            'which', 'that', 'who', 'whom', 'whose', 'where'
        ]
        
        if last_word in incomplete_endings:
            return True
        
        # Check if ends with comma (likely incomplete)
        if text.endswith(','):
            return True
        
        return False
    
    def get_line_type(self, text: str) -> str:
        """
        Determine if line is likely a question, answer, or other.
        Used to intelligently merge across page breaks.
        
        Args:
            text: Text line to classify
            
        Returns:
            'question', 'answer', or 'unknown'
        """
        text_stripped = text.strip()
        
        # Question patterns
        question_patterns = [
            r'^q[\s\.\(]',  # Q., Q , Q(
            r'^ques[\s\.]',  # Ques., Ques
            r'^question[\s\.]',  # Question.
            r'^\d+\.\s*[qQ]',  # 1. Q
        ]
        
        for pattern in question_patterns:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return 'question'
        
        # Question mark at end
        if '?' in text_stripped:
            return 'question'
        
        # Answer patterns
        answer_patterns = [
            r'^ans[\s\.\:]',  # Ans., Ans:
            r'^answer[\s\.\:]',  # Answer:
            r'^a[\s\.\:]',  # A., A:
            r'^a\d+[\s\.]',  # A1., A2.
        ]
        
        for pattern in answer_patterns:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return 'answer'
        
        return 'unknown'
    
    def minimal_clean(self, text: str) -> str:
        """
        Minimal cleaning to preserve OCR characteristics for DeBERTa.
        
        Your DeBERTa model was trained on OCR data with artifacts,
        so we keep most of the original text structure and only:
        - Remove null bytes
        - Normalize excessive whitespace
        - Strip leading/trailing whitespace
        
        Args:
            text: Raw extracted text
            
        Returns:
            Minimally cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize multiple spaces to single space (but preserve newlines)
        text = re.sub(r' +', ' ', text)
        
        # Remove excessive newlines (keep at most 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text for DeBERTa classification.
        
        Args:
            text: Raw extracted text (possibly with DeepSeek annotations)
            
        Returns:
            Cleaned text ready for line extraction
        """
        # Remove DeepSeek bounding boxes if enabled
        if self.remove_bounding_boxes:
            text = self.remove_deepseek_annotations(text)
        
        # Apply minimal cleaning to preserve OCR characteristics
        text = self.minimal_clean(text)
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def should_merge_pages(self, page1_text: str, page2_text: str) -> bool:
        """
        Determine if two consecutive pages should be merged based on content similarity.
        
        Args:
            page1_text: Text from first page
            page2_text: Text from second page
            
        Returns:
            True if pages should be merged, False otherwise
        """
        # Check for continuation indicators
        continuation_indicators = [
            page1_text.endswith(('-', 'contin', 'cont.')),
            not page1_text.endswith(('.', '!', '?')),
            page2_text[0].islower() if page2_text else False
        ]
        
        # Calculate text similarity
        similarity = self.calculate_similarity(page1_text[-200:], page2_text[:200])
        
        # Merge if high similarity or continuation indicators present
        return similarity > self.similarity_threshold or any(continuation_indicators)
    
    def merge_pages(self, texts: List[str]) -> List[str]:
        """
        Merge related pages based on content similarity and continuation.
        
        Args:
            texts: List of text strings from multiple pages
            
        Returns:
            List of merged text segments
        """
        if not texts:
            return []
        
        merged_texts = []
        current_text = texts[0]
        
        for i in range(1, len(texts)):
            if self.should_merge_pages(current_text, texts[i]):
                # Merge with current text
                separator = '' if current_text.endswith('-') else ' '
                current_text = current_text.rstrip('-') + separator + texts[i]
                logger.info(f"Merged page {i} with previous page")
            else:
                # Save current and start new
                merged_texts.append(current_text)
                current_text = texts[i]
        
        # Add the last segment
        merged_texts.append(current_text)
        
        return merged_texts
    
    def process_texts(self, texts: List[str]) -> List[str]:
        """
        Complete text processing pipeline: clean and merge.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of processed and merged text segments
        """
        logger.info(f"Processing {len(texts)} text segments")
        
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Merge related pages
        merged_texts = self.merge_pages(cleaned_texts)
        
        logger.info(f"Processed into {len(merged_texts)} merged segments")
        return merged_texts
    
    def merge_page_breaks(self, lines: List[str]) -> List[str]:
        """
        Intelligently merge lines that were split by page breaks.
        
        Strategy:
        1. Identify and remove page markers
        2. Check if line before page break is incomplete
        3. Check if line after page break continues the SAME item (Q or A)
        4. DON'T merge if switching from Question to Answer or vice versa
        5. Merge only if both conditions met
        
        KEY RULE: Never merge across Q->A or A->Q boundaries!
        
        Args:
            lines: List of text lines with potential page breaks
            
        Returns:
            List of merged lines
        """
        merged_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Skip empty lines
            if not current_line:
                i += 1
                continue
            
            # Skip page markers (don't add to output)
            if self.is_page_marker(current_line):
                logger.debug(f"Removing page marker: {current_line}")
                i += 1
                continue
            
            # Look ahead for potential merge
            merge_with = []
            current_type = self.get_line_type(current_line)
            
            # Check if current line is incomplete
            if self.is_incomplete_sentence(current_line):
                # Look at next few lines
                j = i + 1
                while j < len(lines) and j < i + 5:  # Look ahead max 5 lines
                    next_line = lines[j].strip()
                    
                    # Skip empty lines and page markers
                    if not next_line or self.is_page_marker(next_line):
                        j += 1
                        continue
                    
                    # Check if next line starts a NEW Q/A (has a label prefix)
                    has_qa_prefix = re.match(
                        r'^(q[\s\.\(]|ques[\s\.]|question[\s\.]|ans[\s\.\:]|answer[\s\.\:]|a[\s\.\:]|a\d+[\s\.])',
                        next_line,
                        re.IGNORECASE
                    )
                    
                    if has_qa_prefix:
                        # Next line starts a new Q or A - DON'T MERGE
                        logger.debug(f"Not merging: next line has Q/A prefix")
                        break
                    
                    # Get next line type
                    next_type = self.get_line_type(next_line)
                    
                    # CRITICAL CHECK: Don't merge if types switch
                    if current_type != 'unknown' and next_type != 'unknown':
                        if current_type != next_type:
                            logger.debug(f"Not merging: type switch from {current_type} to {next_type}")
                            break
                    
                    # Safe to merge: no prefix, compatible types
                    merge_with.append(next_line)
                    j += 1
                    
                    # Continue looking if this merged line is also incomplete
                    if not self.is_incomplete_sentence(next_line):
                        break
            
            # Perform merge or add as-is
            if merge_with:
                merged_text = current_line + ' ' + ' '.join(merge_with)
                merged_lines.append(merged_text)
                logger.debug(f"Merged {len(merge_with) + 1} lines into: {merged_text[:50]}...")
                i += len(merge_with) + 1
            else:
                merged_lines.append(current_line)
                i += 1
        
        logger.info(f"Page break merging: {len(lines)} → {len(merged_lines)} lines")
        return merged_lines
    
    def extract_lines_from_markdown(self, markdown_text: str, merge_breaks: bool = True) -> List[str]:
        """
        Extract individual text lines from DeepSeek markdown output with intelligent page break handling.
        
        This matches the format of your DeBERTa training data:
        - Questions starting with "Q", "Question", "Ques", etc.
        - Answers starting with "Ans", "Answer", "A", etc.
        - Other lines (headers, page numbers, dates, etc.)
        
        Args:
            markdown_text: Raw markdown text from DeepSeek OCR
            merge_breaks: Whether to merge lines split by page breaks (default: True)
            
        Returns:
            List of individual text lines
        """
        # First clean the text (removes bounding boxes)
        cleaned_text = self.clean_text(markdown_text)
        
        # Split by newlines
        lines = cleaned_text.split('\n')
        
        # Process each line with minimal cleaning
        processed_lines = []
        for line in lines:
            line = self.minimal_clean(line)
            if line:  # Only keep non-empty lines
                processed_lines.append(line)
        
        # Merge page breaks if enabled
        if merge_breaks:
            processed_lines = self.merge_page_breaks(processed_lines)
            logger.info(f"Extracted {len(processed_lines)} lines from markdown (with page break merging)")
        else:
            logger.info(f"Extracted {len(processed_lines)} lines from markdown (no page break merging)")
        
        return processed_lines
    
    def lines_to_jsonl(self, lines: List[str], output_file: str = None) -> Tuple[List[Dict], Dict]:
        """
        Convert lines to JSONL format matching DeBERTa training data with statistics.
        
        Format: {"text": "...", "label": ""}
        
        Labels are empty initially - DeBERTa model will predict them as:
        - "question"
        - "answer" 
        - "other"
        
        Args:
            lines: List of text lines
            output_file: Optional path to save JSONL file
            
        Returns:
            Tuple of (jsonl_data, statistics)
        """
        jsonl_data = []
        stats = {
            'total_lines': len(lines),
            'likely_questions': 0,
            'likely_answers': 0,
            'other': 0
        }
        
        for line in lines:
            # Create entry without label (model will predict)
            entry = {
                "text": line,
                "label": ""  # Empty - to be predicted by DeBERTa
            }
            jsonl_data.append(entry)
            
            # Update statistics
            line_type = self.get_line_type(line)
            if line_type == 'question':
                stats['likely_questions'] += 1
            elif line_type == 'answer':
                stats['likely_answers'] += 1
            else:
                stats['other'] += 1
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in jsonl_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(jsonl_data)} lines to {output_file}")
            logger.info(f"Stats - Q: {stats['likely_questions']}, A: {stats['likely_answers']}, Other: {stats['other']}")
        
        return jsonl_data, stats
    
    def process_for_classification(self, text: str, output_jsonl: str = None, 
                                  merge_page_breaks: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Complete pipeline: DeepSeek markdown → line-by-line JSONL for DeBERTa.
        
        Args:
            text: Raw DeepSeek markdown text
            output_jsonl: Optional path to save JSONL output
            merge_page_breaks: Whether to merge lines split by page breaks (default: True)
            
        Returns:
            Tuple of (jsonl_data, statistics)
        """
        logger.info("Processing DeepSeek markdown for DeBERTa classification")
        
        # Extract individual lines with optional page break merging
        lines = self.extract_lines_from_markdown(text, merge_breaks=merge_page_breaks)
        
        # Convert to JSONL format with statistics
        jsonl_data, stats = self.lines_to_jsonl(lines, output_jsonl)
        
        logger.info(f"Prepared {len(jsonl_data)} entries for classification")
        return jsonl_data, stats
    
    def split_into_chunks(self, text: str, max_length: int = 512, 
                         overlap: int = 50) -> List[str]:
        """
        Split long text into overlapping chunks for model input.
        
        Args:
            text: Input text to split
            max_length: Maximum length of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def analyze_lines(self, jsonl_data: List[Dict]) -> Dict:
        """
        Analyze extracted lines for verification.
        
        Args:
            jsonl_data: List of dictionaries with text/label
            
        Returns:
            Dictionary with analysis statistics
        """
        if not jsonl_data:
            return {}
        
        lengths = [len(entry['text']) for entry in jsonl_data]
        
        analysis = {
            'total_lines': len(jsonl_data),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'samples': jsonl_data[:5]  # First 5 samples
        }
        
        logger.info(f"Analysis: {analysis['total_lines']} lines, "
                   f"avg length {analysis['avg_length']:.1f} chars")
        
        return analysis
    
    # =========================================================================
    # CSV EXPORT METHODS
    # =========================================================================
    
    def export_to_csv(self, texts: List[str], output_path: str, 
                      include_metadata: bool = True) -> None:
        """
        Process OCR texts and save to CSV for DeBERTa classification.
        
        Args:
            texts: List of raw OCR extracted texts (one per document)
            output_path: Path to output CSV file
            include_metadata: Whether to include metadata columns
        """
        all_segments = []
        
        for doc_idx, raw_text in enumerate(texts):
            segments = self.extract_text_segments(raw_text)
            
            for segment in segments:
                segment['document_id'] = doc_idx + 1
                all_segments.append(segment)
        
        # Write to CSV
        if all_segments:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if include_metadata:
                fieldnames = ['document_id', 'id', 'text', 'char_count', 'word_count', 'likely_type']
            else:
                fieldnames = ['text']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for segment in all_segments:
                    if include_metadata:
                        writer.writerow(segment)
                    else:
                        writer.writerow({'text': segment['text']})
            
            logger.info(f"Processed {len(texts)} documents")
            logger.info(f"Extracted {len(all_segments)} text segments")
            logger.info(f"Saved to: {output_path}")
        else:
            logger.warning("No segments extracted from input texts")
    
    def process_single_document(self, raw_text: str, aggressive: bool = False) -> str:
        """
        Quick function to preprocess a single document.
        
        Args:
            raw_text: Raw OCR text from DeepSeek OCR
            aggressive: Apply aggressive OCR corrections
            
        Returns:
            Preprocessed text ready for classification
        """
        return self.preprocess_ocr_text(raw_text, aggressive=aggressive)
    
    def full_pipeline(self, text: str, output_jsonl: str = None, 
                     output_csv: str = None, merge_page_breaks: bool = True,
                     aggressive_ocr: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Complete processing pipeline: OCR cleanup → Page break handling → 
        Line extraction → Export
        
        Args:
            text: Raw DeepSeek OCR markdown text
            output_jsonl: Optional path to save JSONL output
            output_csv: Optional path to save CSV output
            merge_page_breaks: Whether to merge lines split by page breaks
            aggressive_ocr: Apply aggressive OCR corrections
            
        Returns:
            Tuple of (jsonl_data, statistics)
        """
        logger.info("Starting full text processing pipeline")
        
        # Step 1: Apply OCR preprocessing
        if aggressive_ocr:
            text = self.preprocess_ocr_text(text, aggressive=True)
            logger.info("Applied aggressive OCR corrections")
        
        # Step 2: Process for classification (handles bounding boxes, page breaks, etc.)
        jsonl_data, stats = self.process_for_classification(
            text, 
            output_jsonl=output_jsonl,
            merge_page_breaks=merge_page_breaks
        )
        
        # Step 3: Export to CSV if requested
        if output_csv:
            self.export_to_csv([text], output_csv, include_metadata=True)
        
        logger.info(f"Pipeline complete: {stats['total_lines']} lines processed")
        return jsonl_data, stats


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_ocr_text(raw_text: str, aggressive: bool = False) -> str:
    """
    Quick function to preprocess OCR text.
    
    Args:
        raw_text: Raw OCR text from DeepSeek OCR
        aggressive: Apply aggressive OCR corrections
        
    Returns:
        Preprocessed text
    """
    processor = TextProcessor(aggressive_ocr_correction=aggressive)
    return processor.preprocess_ocr_text(raw_text)


def process_to_jsonl(text: str, output_path: str, merge_breaks: bool = True) -> List[Dict]:
    """
    Process OCR text and save to JSONL format.
    
    Args:
        text: Raw OCR text
        output_path: Path to save JSONL file
        merge_breaks: Merge page breaks
        
    Returns:
        List of processed entries
    """
    processor = TextProcessor()
    jsonl_data, stats = processor.process_for_classification(
        text, 
        output_jsonl=output_path,
        merge_page_breaks=merge_breaks
    )
    return jsonl_data


def process_to_csv(texts: List[str], output_path: str, include_metadata: bool = True) -> None:
    """
    Process multiple OCR texts and save to CSV.
    
    Args:
        texts: List of raw OCR texts
        output_path: Path to save CSV file
        include_metadata: Include metadata columns
    """
    processor = TextProcessor()
    processor.export_to_csv(texts, output_path, include_metadata=include_metadata)
