"""
Qwen3-4B-Instruct-2507 based Zero-Shot classifier for OCR text classification.
Uses only a detailed system prompt without any examples.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import re
import json
import logging
from datetime import datetime

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


class ZeroShotClassifier:
    """
    Zero-shot classifier using Qwen3-4B-Instruct-2507 for OCR text classification.
    
    Features:
    - No examples needed - relies on detailed system prompt
    - Intelligent line merging to handle multi-line Q&A content
    - Pattern-based pre-classification for obvious cases (optional)
    """
    
    # Detailed zero-shot system prompt
    SYSTEM_PROMPT = """You are an expert text classifier for academic documents extracted via OCR. Your task is to classify each piece of text into exactly ONE of three categories.

## CATEGORIES

### **question**
Text that asks for information, solutions, or explanations:
- Starts with question markers: Q, Ques, Question, Q1, Q-1, etc.
- Interrogative sentences starting with: What, Why, How, Who, Where, When, Which
- Imperative requests: Explain, Describe, Define, Differentiate, Discuss, Compare
- Problem statements or scenarios requiring solutions
- Text ending with "?" that seeks an answer

### **answer**
Text that provides information, solutions, or explanations:
- Starts with answer markers: Ans, Answer, Solution, A1, A-1, etc.
- Direct responses to questions (factual statements, explanations)
- Technical explanations, definitions, or descriptions
- Narrative responses describing events or situations
- Numbered or bulleted points explaining concepts (e.g., "1) Matter is made up of particles")
- Properties, characteristics, or features being listed
- Any explanatory content that teaches or informs

### **metadata**
Administrative, structural, or non-content text:
- Page markers: Page No., --- PAGE 1 ---, Page 5 of 10
- Single words that are headers: Date, Name, DOMS, Notes
- Section headers without content: "QUESTIONS:", "Short questions", "Chapter 1"
- Student/submission info: Names, Roll numbers, IDs, Dates
- Website URLs, watermarks, branding
- Very short fragments (1-2 words) that are labels

## CRITICAL RULES

1. **Content over markers**: A numbered point like "1) Particles have space between them" is an ANSWER (fact/explanation), not a question.

2. **Sentence structure matters**: 
   - "What is photosynthesis?" → question (interrogative)
   - "Photosynthesis is the process..." → answer (declarative/explanatory)

3. **Section headers are metadata**: "QUESTIONS:", "→ Evaporation:", "Long Answer Questions" are metadata (headers), not actual questions.

4. **Explicit markers override semantics**: "Ans: What happened?" is an ANSWER because it starts with "Ans:".

5. **Educational content is usually answer**: In study notes, explanations of concepts, properties, and facts are answers being taught to students.

## OUTPUT FORMAT
Reply with ONLY one word: question, answer, or metadata"""

    # Patterns for quick pre-classification (optional, high confidence)
    QUESTION_PATTERNS = [
        r'^Q\s*[\d\-\.:]+',             # Q1, Q-1, Q.1, Q:1
        r'^Ques[\s\-\.:]',              # Ques, Ques-, Ques.
        r'^Question\s*[\d\-\.:]*',      # Question, Question 1
        r'^(What|Why|How|Who|Where|When|Which)\s+.+\?$',  # Full interrogative sentences ending with ?
    ]
    
    ANSWER_PATTERNS = [
        r'^Ans\s*[\-\.:]+',             # Ans-, Ans., Ans:
        r'^Answer\s*[\-\.:]*',          # Answer, Answer:
        r'^Solution[\s\-\.:]*',         # Solution, Solution:
    ]
    
    METADATA_PATTERNS = [
        r'^---\s*PAGE\s*\d+\s*---',     # --- PAGE 1 ---
        r'^Page\s*(No\.?|Number)?',     # Page No., Page Number
        r'^\d+\s*(of|/)\s*\d+$',        # 5 of 10, 5/10
        r'^(Name|Roll|ID|Student)\s*[:.]', # Name:, Roll No:
        r'^\d{2}[A-Z]{2,3}\d{4,}',      # 23BEC1025 (student ID pattern)
        r'^(Date|Due|Submitted)[\s:]*', # Date:, Due Date:
        r'^[A-Z]{2,5}$',                # Short all-caps (DOMS, AWS)
        r'^(www\.|http|@)',             # URLs and handles
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_pattern_matching: bool = True
    ):
        """
        Initialize the Zero-shot Qwen classifier.
        
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
        
        logger.info(f"Loading Zero-shot Qwen classifier: {model_name}")
        logger.info(f"Device: {self.device}, dtype: {torch_dtype}")
        
        print(f"Loading {model_name} (Zero-shot mode)...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        
        self.model.eval()
        logger.info("Zero-shot Qwen classifier loaded successfully")
        print("Model loaded successfully!")
    
    def _check_patterns(self, text: str) -> Optional[str]:
        """
        Check if text matches any pre-defined patterns for quick classification.
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
    
    def _classify_single(self, text: str) -> Tuple[str, float, str]:
        """
        Classify a single text using zero-shot LLM.
        
        Returns:
            Tuple of (label, confidence, reasoning)
        """
        # Skip empty or very short text
        if not text or len(text.strip()) < 2:
            return "unknown", 0.0, "Text too short or empty"
        
        # Try pattern matching first if enabled
        if self.use_pattern_matching:
            pattern_result = self._check_patterns(text)
            if pattern_result:
                return pattern_result, 0.98, "Pattern matched: " + pattern_result
        
        # Use LLM for classification
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this text:\n\n{text}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()
        
        # Parse response
        if "question" in response:
            return "question", 0.9, "LLM classified as question"
        elif "answer" in response:
            return "answer", 0.9, "LLM classified as answer"
        elif "metadata" in response:
            return "metadata", 0.9, "LLM classified as metadata"
        else:
            return "unknown", 0.5, f"LLM response unclear: {response}"
    
    def _merge_related_lines(self, lines: List[str]) -> List[Tuple[str, List[int]]]:
        """
        Merge related lines that belong to the same question or answer.
        """
        if not lines:
            return []
        
        merged_blocks = []
        current_block = []
        current_indices = []
        
        # Patterns that indicate a new block
        new_block_patterns = [
            r'^Q\s*[\d\-\.\:]',
            r'^Ques[\s\-\.\:]',
            r'^Question\s*\d*',
            r'^Ans[\s\-\.\:]',
            r'^Answer[\s\-\.\:]',
            r'^---\s*PAGE',
            r'^\([a-z]\)',
            r'^\d+\s*[\.\)]',
        ]
        
        def is_new_block_start(text: str) -> bool:
            for pattern in new_block_patterns:
                if re.match(pattern, text.strip(), re.IGNORECASE):
                    return True
            return False
        
        def is_continuation(text: str, prev_text: str) -> bool:
            text = text.strip()
            if not text:
                return False
            
            # Check if previous text ends with sentence terminator
            if prev_text and prev_text.rstrip()[-1] in '.?!':
                # Previous sentence complete - less likely to be continuation
                # But short fragments might still be continuations
                if len(text) < 50 and text[0].islower():
                    return True
                return False
            
            # Starts with lowercase - likely continuation
            if text[0].islower():
                return True
            
            # Short text without markers - might be continuation
            if len(text) < 30 and not is_new_block_start(text):
                return True
            
            return False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            if not current_block:
                current_block.append(line)
                current_indices.append(i)
            elif is_new_block_start(line):
                # Save current block and start new one
                merged_blocks.append((" ".join(current_block), current_indices.copy()))
                current_block = [line]
                current_indices = [i]
            elif is_continuation(line, current_block[-1] if current_block else ""):
                # Merge with current block
                current_block.append(line)
                current_indices.append(i)
            else:
                # Save current block and start new one
                merged_blocks.append((" ".join(current_block), current_indices.copy()))
                current_block = [line]
                current_indices = [i]
        
        # Don't forget the last block
        if current_block:
            merged_blocks.append((" ".join(current_block), current_indices.copy()))
        
        return merged_blocks
    
    def classify_document(
        self,
        lines: List[str],
        merge_lines: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Classify all lines in a document using zero-shot classification.
        
        Args:
            lines: List of text lines to classify
            merge_lines: Whether to merge related lines before classification
            show_progress: Whether to show progress during classification
            
        Returns:
            Dictionary with questions, answers, metadata lists and statistics
        """
        # Merge lines if requested
        if merge_lines:
            print(f"  Merging into logical blocks...")
            blocks = self._merge_related_lines(lines)
            print(f"  Created {len(blocks)} blocks for classification")
        else:
            blocks = [(line, [i]) for i, line in enumerate(lines) if line.strip()]
        
        results = {
            "questions": [],
            "answers": [],
            "metadata": [],
            "unknown": []
        }
        
        for i, (text, indices) in enumerate(blocks):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(blocks)} blocks classified...")
            
            label, confidence, reasoning = self._classify_single(text)
            
            result = {
                "text": text,
                "confidence": confidence,
                "reasoning": reasoning,
                "original_line_indices": indices
            }
            
            if label == "question":
                results["questions"].append(result)
            elif label == "answer":
                results["answers"].append(result)
            elif label == "metadata":
                results["metadata"].append(result)
            else:
                results["unknown"].append(result)
        
        # Add statistics
        results["statistics"] = {
            "total_original_lines": len(lines),
            "total_merged_blocks": len(blocks),
            "questions_count": len(results["questions"]),
            "answers_count": len(results["answers"]),
            "metadata_count": len(results["metadata"]),
            "unknown_count": len(results["unknown"])
        }
        
        results["model"] = self.model_name
        results["method"] = "zero-shot-llm"
        
        return results
