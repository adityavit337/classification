"""Text processing for DeepSeek OCR markdown output to DeBERTa classification format."""

import re
import json
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Process DeepSeek OCR markdown output for DeBERTa classification.
    
    Converts markdown with bounding boxes to clean line-by-line format
    matching the DeBERTa training data structure.
    """
    
    def __init__(self, similarity_threshold: float = 0.8, 
                 remove_bounding_boxes: bool = True):
        """
        Initialize text processor.
        
        Args:
            similarity_threshold: Threshold for determining if pages should be merged (0-1)
            remove_bounding_boxes: Remove DeepSeek bounding box annotations (default: True)
        """
        self.similarity_threshold = similarity_threshold
        self.remove_bounding_boxes = remove_bounding_boxes
        
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
