"""
Qwen3-4B-Instruct-2507 based classifier for OCR text classification.
Uses few-shot learning with explicit examples to classify text as question/answer/metadata.
Includes intelligent line merging to handle multi-line Q&A content.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import re
import json
import logging

logger = logging.getLogger(__name__)


class TextType(Enum):
    """Classification categories for OCR text."""
    QUESTION = "question"
    ANSWER = "answer"
    METADATA = "metadata"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of classifying a single text line."""
    text: str
    predicted_type: TextType
    confidence: float
    reasoning: str


class QwenClassifier:
    """
    Few-shot classifier using Qwen3-4B-Instruct-2507 for OCR text classification.
    
    Features:
    - Intelligent line merging to handle multi-line questions/answers
    - Enhanced few-shot examples covering edge cases
    - Pattern-based pre-classification for obvious cases
    """
    
    # Enhanced few-shot examples with more variety and edge cases
    FEW_SHOT_EXAMPLES = [
        # Questions with explicit markers
        {"text": "Ques-1: You are the lead architect for a popular e-commerce platform. The business has announced a massive 'flash sale' event that is expected to increase website traffic by 10x.", "label": "question"},
        {"text": "Q1- The Buddha Said, \"The world is affected with death and decay, therefore the wise do not grieve.\"", "label": "question"},
        {"text": "Q (a) Who is 'he' in the above extract? How did he feel when he saw the__(thing)?", "label": "question"},
        {"text": "Question 2: Design a scalable AWS architecture for handling media uploads from mobile devices.", "label": "question"},
        
        # Questions without markers (interrogative)
        {"text": "What are the key components needed to handle 10x traffic increase during flash sales?", "label": "question"},
        {"text": "How would you implement auto-scaling for the database layer?", "label": "question"},
        {"text": "Why is caching important in high-traffic scenarios?", "label": "question"},
        {"text": "Explain the difference b/w photosynthesis and symbioses.", "label": "question"},
        {"text": "Differentiate b/w mitosis & meiosis.", "label": "question"},
        {"text": "Determine the difference b/w North & South?", "label": "question"},
        
        # Problem statements/scenarios (questions)
        {"text": "The current architecture relies on a single EC2 instance for the web server and a single RDS instance. This will fail under heavy load. Design a solution.", "label": "question"},
        {"text": "During high demand events like flash sales, e-commerce platforms experience massive traffic spikes. Your task is to redesign the architecture.", "label": "question"},
        
        # Answers with explicit markers
        {"text": "Ans: Multi-AZ redundancy, auto-scaling groups, ElastiCache for caching, and decoupling services using SQS.", "label": "answer"},
        {"text": "Ans-2: Scalable and cost-effective AWS architecture for Media upload includes S3, CloudFront, and Lambda.", "label": "answer"},
        {"text": "Answer: 'Precisely' is the synonym of exactly. The author uses this word to emphasize accuracy.", "label": "answer"},
        {"text": "Ans: The pirate that came to rob them was actually a kind-hearted sailor in disguise.", "label": "answer"},
        
        # Answers WITHOUT explicit markers (statements that respond to questions)
        {"text": "Photosynthesis is good for plant while symbioses is not.", "label": "answer"},
        {"text": "At North is north & south is south.", "label": "answer"},
        {"text": "Mitosis produces identical diploid cells used for growth & repair.", "label": "answer"},
        {"text": "The cell is the basic unit of life.", "label": "answer"},
        
        # Technical explanations (answers without markers)
        {"text": "Application Layer: Amazon ECS or EKS containers behind an Application Load Balancer provide horizontal scaling.", "label": "answer"},
        {"text": "Database Layer: Amazon Aurora with read replicas and ElastiCache Redis for session management.", "label": "answer"},
        {"text": "The solution uses Amazon S3 for storage, CloudFront for CDN, and Lambda for serverless processing.", "label": "answer"},
        {"text": "Auto Scaling Groups automatically adjust EC2 instance count based on CPU utilization metrics.", "label": "answer"},
        
        # Narrative answers (story-based responses without markers)
        # IMPORTANT: "When X happened, Y felt..." is a narrative answer, NOT a question
        {"text": "When Valli saw the dead cow by the roadside, she was deeply saddened and felt a pang of disappointment.", "label": "answer"},
        {"text": "When he opened the door, he was surprised to find the room empty.", "label": "answer"},
        {"text": "When the teacher asked, the student confidently answered the question.", "label": "answer"},
        {"text": "During her first journey, Valli experienced the thrill of seeing the world outside her village.", "label": "answer"},
        {"text": "She enjoyed the sights of the canal, the palm trees, the mountains, and the green fields.", "label": "answer"},
        {"text": "He felt a sense of wonder and amazement when he first saw the grand palace.", "label": "answer"},
        {"text": "The sight of the lifeless cow made her reflect on the nature of life and death.", "label": "answer"},
        
        # Single-sentence factual answers
        {"text": "Southwest monsoon winds account for rainfall along the Malabar coast.", "label": "answer"},
        {"text": "The Thar Desert experiences the highest diurnal range of temperature in India.", "label": "answer"},
        {"text": "Jet streams are narrow belts of high altitude winds in the troposphere.", "label": "answer"},
        {"text": "The seasonal reversal in wind direction during a year is called the monsoon.", "label": "answer"},
        
        # Definition and explanation answers
        {"text": "These are narrow belts of high altitude (above 12,000 m) in the troposphere.", "label": "answer"},
        {"text": "This is because it is filled with sand which gets heated up quickly during day and cooled very quickly during nights.", "label": "answer"},
        {"text": "Monsoon tends to have 'breaks' in rainfall which means there are wet and dry spells in between.", "label": "answer"},
        
        # Metadata - ONLY short words/phrases (max 3 words)
        {"text": "DOMS", "label": "metadata"},
        {"text": "Page No.", "label": "metadata"},
        {"text": "Date", "label": "metadata"},
        {"text": "Name:", "label": "metadata"},
        {"text": "Roll No:", "label": "metadata"},
        {"text": "Section A", "label": "metadata"},    
        {"text": "--- PAGE 1 ---", "label": "metadata"},
        {"text": "Total Marks:", "label": "metadata"},
    ]
    
    # Patterns for quick pre-classification (high confidence)
    QUESTION_PATTERNS = [
        r'^Q\s*[\d\-\.:]+',             # Q1, Q-1, Q.1, Q:1
        r'^Q\s+[A-Z][a-z]',             # Q What, Q How, Q Why
        r'^Ques[\s\-\.:]',              # Ques, Ques-, Ques.
        r'^Question\s*[\d\-\.:]*',      # Question, Question 1
        r'^\(\s*[a-z]\s*\)',            # (a), (b), etc.
        r'^\d+\s*[\.\)]\s*[A-Z]',       # 1. What, 2) How
        r'^(What|Why|How|Who|Where|When|Which|Explain|Describe|Define|Determine|Differentiate)\s', # Interrogative starts
    ]
    
    ANSWER_PATTERNS = [
        r'^Ans\s',                      # Ans followed by space
        r'^Ans[\-\.:]+',                # Ans-, Ans., Ans:
        r'^Answer\s*[\d\-\.:]*',        # Answer, Answer:, Answer 2
        r'^Solution[\s\-\.:]*',         # Solution, Solution:
        r'^A[\s]*[\d]+[\s\-\.:]+',      # A1, A-1, A.1
    ]
    
    METADATA_PATTERNS = [
        r'^---\s*PAGE\s*\d+\s*---',    # --- PAGE 1 ---
        r'^Page\s*(No\.?|Number)?',    # Page No., Page Number
        r'^\d+\s*(of|/)\s*\d+$',       # 5 of 10, 5/10
        r'^(Section|Part)\s*[A-Z]\s*[:.]?\s*$',  # Section A, Part 1 (must be short header)
        r'^(Section|Part)\s*[A-Z]\s*[:.-]',  # Section A:, Part B-
        r'^(Name|Roll|ID|Student)\s*[:.]', # Name:, Roll No:
        r'^\d{2}[A-Z]{2,3}\d{4,}',     # 23BEC1025 (student ID pattern)
        r'^(Date|Due|Submitted)[\s:]*', # Date:, Due Date:
        r'^Total\s*(Marks|Points)',    # Total Marks
        r'^[A-Z]{2,5}$',               # Short all-caps (DOMS, AWS)
        r'^(x\s*)+[xo]?\s*$',          # Tic-tac-toe patterns like "x x x"
        r'^[xo]\s+[xo]\s+[xo]$',       # More tic-tac-toe
        r'^(Short|Long|Very Short|Medium)\s+(Answer\s+)?(type\s+)?Questions?\s*$',  # Section headers like "Long Answer type Questions"
    ]
    
    # System prompt for classification
    SYSTEM_PROMPT = """You are an expert OCR text classifier for academic documents. Classify each text into exactly one category:

**question**: Problem statements, queries, exercises, or scenarios requiring a solution.
- Text starting with Q, Ques, Question markers (Q1, Ques-1, etc.)
- Interrogative sentences (What, Why, How, Who, Where, When)
- Problem scenarios that need to be solved or analyzed
- Case study descriptions asking for solutions

**answer**: Solutions, explanations, responses, or technical details.
- Text starting with Ans, Answer, Solution markers
- Technical explanations with specific technologies/services
- Direct responses or explanations to questions
- Implementation details or architecture descriptions

**metadata**: Administrative information, headers, footers, or structural elements.
- Student names, IDs, roll numbers
- Page numbers, dates, submission info
- Section headers/titles (like "Short questions", "Section A")
- Organization names, watermarks, assignment titles
- Short ALL-CAPS text (usually headers)

CRITICAL RULES:
1. Explicit markers OVERRIDE semantic content (Ans: followed by a question-like text → answer)
2. Section titles like "Short questions" are METADATA, not questions
3. Problem scenarios without question marks are still QUESTIONS if they ask for solutions
4. Technical layer descriptions (Application Layer:, Database Layer:) are ANSWERS

Reply with ONLY one word: question, answer, or metadata."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_pattern_matching: bool = True
    ):
        """
        Initialize the Qwen classifier.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
            torch_dtype: Data type for model weights
            use_pattern_matching: Whether to use regex patterns for obvious cases
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.use_pattern_matching = use_pattern_matching
        
        logger.info(f"Loading Qwen classifier: {model_name}")
        logger.info(f"Device: {self.device}, dtype: {torch_dtype}")
        
        print(f"Loading {model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        
        self.model.eval()
        logger.info("Qwen classifier loaded successfully")
        print("Model loaded successfully!")
    
    def _check_patterns(self, text: str) -> Optional[str]:
        """
        Check if text matches any pre-defined patterns for quick classification.
        
        Args:
            text: Text to check
            
        Returns:
            Label if pattern matched, None otherwise
        """
        text_stripped = text.strip()
        
        # Check question patterns
        for pattern in self.QUESTION_PATTERNS:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return "question"
        
        # Check answer patterns
        for pattern in self.ANSWER_PATTERNS:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return "answer"
        
        # Check metadata patterns
        for pattern in self.METADATA_PATTERNS:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return "metadata"
        
        return None
    
    def _split_on_qa_markers(self, lines: List[str]) -> List[str]:
        """
        Pre-process lines to split on Q/A markers that appear mid-line.
        
        For example: "Answer is X Q What is Y?" becomes two lines.
        
        Args:
            lines: Original text lines
            
        Returns:
            List of lines with Q/A markers at line starts
        """
        result = []
        
        # Patterns to split on (these indicate new Q or A starting mid-line)
        split_patterns = [
            r'\s+(Q\s*\d*[\.\:\-]?\s+)',           # " Q ", " Q1 ", " Q: "
            r'\s+(Q\s+[A-Z])',                      # " Q What", " Q How"
            r'\s+(Ans\s*[\.\:\-]?\s*)',             # " Ans ", " Ans: "
            r'\s+(Answer\s*\d*[\.\:\-]?\s*)',       # " Answer ", " Answer 2"
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to split on each pattern
            split_made = False
            for pattern in split_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Split at this position
                    before = line[:match.start()].strip()
                    after = line[match.start():].strip()
                    
                    if before:
                        result.append(before)
                    if after:
                        result.append(after)
                    split_made = True
                    break
            
            if not split_made:
                result.append(line)
        
        return result
    
    def _merge_related_lines(self, lines: List[str]) -> List[Tuple[str, List[int]]]:
        """
        Merge related lines that belong to the same question or answer.
        
        Strategy:
        - Lines starting with Q/Ques/Ans markers start new blocks
        - Continuation lines (lowercase start, no markers) merge with previous
        - Short metadata-like lines stay separate
        
        Args:
            lines: List of text lines
            
        Returns:
            List of (merged_text, original_line_indices) tuples
        """
        if not lines:
            return []
        
        merged = []
        current_block = []
        current_indices = []
        
        # Patterns that indicate start of new content block
        new_block_patterns = [
            r'^Q\s*[\d\-\.\:\(]',         # Q1, Q-1, Q(a)
            r'^Q\s+[A-Z]',                 # Q What, Q How (Q followed by word)
            r'^Ques[\s\-\.\:]',            # Ques, Ques-1
            r'^Question\s*[\d\-\.\:]*',    # Question 1
            r'^Ans\s',                     # Ans followed by space
            r'^Ans[\.\:\-]+',              # Ans:, Ans-, Ans.
            r'^Answer[\s\-\.\:]*',         # Answer:
            r'^Solution[\s\-\.\:]*',       # Solution:
            r'^---\s*PAGE',                # Page markers
            r'^\d+\s*[\.\)]\s+[A-Z]',      # Numbered items like "1. What"
            r'^[A-Z][a-z]+\s+Layer:',      # Application Layer:, Database Layer:
        ]
        
        # Patterns that indicate standalone metadata
        standalone_patterns = [
            r'^[A-Z]{2,6}$',             # Short all-caps
            r'^Page\s*(No|Number)?\.?',  # Page markers
            r'^\d+\s*(of|/)\s*\d+$',     # Page numbers
            r'^(Name|Roll|ID)[\s\:]+',   # Name fields
            r'^\d{2}[A-Z]{2,3}\d{4,}',   # Student IDs
            r'^---.*---$',               # Separator lines
        ]
        
        def is_new_block_start(text: str) -> bool:
            text = text.strip()
            for pattern in new_block_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
            # Also check if it's a question (ends with ?) and is substantial
            if text.endswith('?') and len(text) > 20:
                return True
            return False
        
        def is_standalone(text: str) -> bool:
            text = text.strip()
            # Very short lines are usually standalone (but not if ending with ?)
            if len(text) < 12 and not text.endswith('?') and not text.endswith(','):
                return True
            for pattern in standalone_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
            return False
        
        def is_continuation(text: str, prev_text: str = "") -> bool:
            text = text.strip()
            if not text:
                return False
            
            # If previous line ended with a sentence terminator, this is likely NOT a continuation
            # unless it starts with lowercase (indicating broken sentence)
            if prev_text and prev_text.rstrip().endswith(('.', '!', '?')):
                # Only continue if this line clearly continues the sentence
                if not text[0].islower():
                    return False
            
            # Starts with lowercase - definitely a continuation
            if text[0].islower():
                return True
            
            # Starts with common continuation words
            continuation_words = ['the', 'and', 'or', 'but', 'which', 'that', 'this', 'these', 'those', 'for', 'with', 'from', 'to', 'in', 'on', 'at', 'by']
            first_word = text.split()[0].lower().rstrip('.,;:')
            if first_word in continuation_words:
                return True
            return False
        
        def prev_block_text() -> str:
            """Get the last line of current block for context."""
            return current_block[-1] if current_block else ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            if is_standalone(line):
                # Save current block if exists
                if current_block:
                    merged.append((' '.join(current_block), current_indices.copy()))
                    current_block = []
                    current_indices = []
                # Add standalone line
                merged.append((line, [i]))
            
            elif is_new_block_start(line):
                # Save current block if exists
                if current_block:
                    merged.append((' '.join(current_block), current_indices.copy()))
                    current_block = []
                    current_indices = []
                # Start new block
                current_block = [line]
                current_indices = [i]
            
            elif is_continuation(line, prev_block_text()) and current_block:
                # Continue current block
                current_block.append(line)
                current_indices.append(i)
            
            else:
                # Not a continuation - start new block
                if current_block:
                    merged.append((' '.join(current_block), current_indices.copy()))
                current_block = [line]
                current_indices = [i]
        
        # Don't forget the last block
        if current_block:
            merged.append((' '.join(current_block), current_indices.copy()))
        
        return merged
    
    def _build_few_shot_messages(self, text: str) -> List[Dict[str, str]]:
        """
        Build few-shot messages for classification.
        
        Args:
            text: Text to classify
            
        Returns:
            List of message dicts for chat template
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add few-shot examples (use a subset to keep context manageable)
        # Select diverse examples
        selected_examples = [
            self.FEW_SHOT_EXAMPLES[0],   # Question with Ques marker
            self.FEW_SHOT_EXAMPLES[4],   # Question interrogative
            self.FEW_SHOT_EXAMPLES[8],   # Problem scenario question
            self.FEW_SHOT_EXAMPLES[9],   # Answer with Ans marker
            self.FEW_SHOT_EXAMPLES[13],  # Answer technical
            self.FEW_SHOT_EXAMPLES[17],  # Metadata name
            self.FEW_SHOT_EXAMPLES[20],  # Metadata section title
            self.FEW_SHOT_EXAMPLES[24],  # Metadata page marker
        ]
        
        for example in selected_examples:
            messages.append({"role": "user", "content": f"Text: '{example['text']}'"})
            messages.append({"role": "assistant", "content": example["label"]})
        
        # Add the text to classify
        messages.append({"role": "user", "content": f"Text: '{text}'"})
        
        return messages
    
    def _parse_response(self, response: str) -> str:
        """
        Parse the model's response to extract label.
        
        Args:
            response: Raw model output
            
        Returns:
            Label string
        """
        response = response.strip().lower()
        
        # Remove any thinking tags if present
        if '<think>' in response:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # Direct match
        if response in ["question", "answer", "metadata"]:
            return response
        
        # Check first word
        first_word = response.split()[0] if response.split() else ""
        if first_word in ["question", "answer", "metadata"]:
            return first_word
        
        # Check if response contains the label
        if "question" in response:
            return "question"
        elif "answer" in response:
            return "answer"
        elif "metadata" in response:
            return "metadata"
        
        return "unknown"
    
    def classify_single(self, text: str, use_llm: bool = True) -> ClassificationResult:
        """
        Classify a single text.
        
        Args:
            text: Text to classify
            use_llm: Whether to use LLM (if False, only pattern matching)
            
        Returns:
            ClassificationResult with prediction
        """
        # Skip empty or very short text
        if not text or len(text.strip()) < 2:
            return ClassificationResult(
                text=text,
                predicted_type=TextType.UNKNOWN,
                confidence=0.0,
                reasoning="Text too short or empty"
            )
        
        text = text.strip()
        
        # Try pattern matching first (fast path)
        if self.use_pattern_matching:
            pattern_match = self._check_patterns(text)
            if pattern_match:
                type_map = {
                    "question": TextType.QUESTION,
                    "answer": TextType.ANSWER,
                    "metadata": TextType.METADATA,
                }
                return ClassificationResult(
                    text=text,
                    predicted_type=type_map[pattern_match],
                    confidence=0.98,
                    reasoning=f"Pattern matched: {pattern_match}"
                )
        
        if not use_llm:
            return ClassificationResult(
                text=text,
                predicted_type=TextType.UNKNOWN,
                confidence=0.0,
                reasoning="No pattern match and LLM disabled"
            )
        
        # Use LLM for classification
        messages = self._build_few_shot_messages(text)
        
        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Decode response (only the new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse response
        label = self._parse_response(response)
        
        # Convert to TextType
        type_map = {
            "question": TextType.QUESTION,
            "answer": TextType.ANSWER,
            "metadata": TextType.METADATA,
            "unknown": TextType.UNKNOWN
        }
        predicted_type = type_map.get(label, TextType.UNKNOWN)
        
        confidence = 0.90 if predicted_type != TextType.UNKNOWN else 0.3
        
        return ClassificationResult(
            text=text,
            predicted_type=predicted_type,
            confidence=confidence,
            reasoning=f"LLM classified as {label}"
        )
    
    def classify_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[ClassificationResult]:
        """
        Classify a batch of texts.
        
        Args:
            texts: List of texts to classify
            show_progress: Whether to show progress
            
        Returns:
            List of ClassificationResults
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Classifying: {i + 1}/{total}")
                print(f"  Progress: {i + 1}/{total} blocks classified...")
            
            result = self.classify_single(text)
            results.append(result)
        
        return results
    
    def classify_document(
        self,
        lines: List[str],
        merge_lines: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Classify an entire document and return organized results.
        
        Args:
            lines: List of text lines from document
            merge_lines: Whether to merge related lines before classification
            show_progress: Whether to show progress
            
        Returns:
            Dictionary with classified content and statistics
        """
        # Filter non-empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Pre-process: split lines that have Q/A markers mid-line
        print(f"  Pre-processing {len(non_empty_lines)} lines (splitting on Q/A markers)...")
        split_lines = self._split_on_qa_markers(non_empty_lines)
        print(f"  After splitting: {len(split_lines)} lines")
        
        if merge_lines:
            print(f"  Merging into logical blocks...")
            merged_blocks = self._merge_related_lines(split_lines)
            texts_to_classify = [block[0] for block in merged_blocks]
            print(f"  Created {len(texts_to_classify)} blocks for classification")
        else:
            texts_to_classify = split_lines
            merged_blocks = [(line, [i]) for i, line in enumerate(non_empty_lines)]
        
        logger.info(f"Classifying {len(texts_to_classify)} blocks with Qwen3-4B-Instruct")
        
        # Classify all blocks
        results = self.classify_batch(texts_to_classify, show_progress)
        
        # Organize by type
        questions = []
        answers = []
        metadata = []
        unknown = []
        
        for result, (_, indices) in zip(results, merged_blocks):
            entry = {
                "text": result.text,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "original_line_indices": indices
            }
            
            if result.predicted_type == TextType.QUESTION:
                questions.append(entry)
            elif result.predicted_type == TextType.ANSWER:
                answers.append(entry)
            elif result.predicted_type == TextType.METADATA:
                metadata.append(entry)
            else:
                unknown.append(entry)
        
        return {
            "questions": questions,
            "answers": answers,
            "metadata": metadata,
            "unknown": unknown,
            "statistics": {
                "total_original_lines": len(non_empty_lines),
                "total_merged_blocks": len(texts_to_classify),
                "questions_count": len(questions),
                "answers_count": len(answers),
                "metadata_count": len(metadata),
                "unknown_count": len(unknown)
            },
            "model": self.model_name,
            "method": "few-shot-llm-with-merging" if merge_lines else "few-shot-llm"
        }


def main():
    """Test the classifier with sample texts."""
    test_texts = [
        "Ques-1: You are the lead architect for a popular e-commerce platform.",
        "Ans: Multi-AZ redundancy FLASH SCALE Caching Decoupling of services",
        "Dhruw Thapar 23BEC1025",
        "Q (a) Who is 'he' in the above extract? How did he feel",
        "Page No.",
        "Application Layer: Amazon ECS or EKS or AWS Lambda",
        "Short questions",
        "the web server and the single RDS instance, will fail",  # Continuation
    ]
    
    print("=" * 60)
    print("QWEN3-4B-INSTRUCT-2507 CLASSIFIER TEST")
    print("=" * 60)
    
    classifier = QwenClassifier()
    
    for text in test_texts:
        result = classifier.classify_single(text)
        print(f"\nText: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  Label: {result.predicted_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reasoning: {result.reasoning}")


if __name__ == "__main__":
    main()
