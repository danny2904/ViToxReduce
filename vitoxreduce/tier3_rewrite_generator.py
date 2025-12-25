#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 3: Contextual Rewriter
Uses BARTpho (Seq2Seq - Span-guided) to rewrite sentences
One-time Rewriting with Beam Search to select Top-1 best result

Refactored to use BartphoSpanRewriter from baseline to ensure identical prompt and logic
"""

import logging
import re
from typing import List, Optional
import torch

from .bartpho_span_baseline import BartphoSpanRewriter, PredictionCleaner

logger = logging.getLogger(__name__)


class ContextualRewriter:
    """Tier 3: Contextual Rewriter - BARTpho Seq2Seq with Span-guided prompt
    
    Uses BartphoSpanRewriter from baseline to ensure identical prompt and logic as baseline
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256,  # Same as baseline default
        num_beams: int = 5
    ):
        """
        Args:
            model_path: Path to fine-tuned BARTpho model
            device: Device to run model on
            max_length: Maximum sequence length (default: 256, same as baseline)
            num_beams: Number of beams for beam search (to select best top-1)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.num_beams = num_beams
        self.model_path = model_path
        
        # Sử dụng BartphoSpanRewriter từ baseline
        logger.info(f"Initializing Contextual Rewriter with BartphoSpanRewriter...")
        self.rewriter = BartphoSpanRewriter(
            model_path=model_path,
            device=self.device,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1,  # Chỉ lấy top-1
            do_sample=False,
        )
        
        # Sử dụng PredictionCleaner từ baseline để clean output
        self.cleaner = PredictionCleaner(enable=True)
        
        logger.info("Contextual Rewriter initialized successfully!")
    
    def rewrite(
        self,
        text: str,
        span_texts: Optional[List[str]] = None,
        spans_indices: Optional[List[List[int]]] = None
    ) -> str:
        """
        Rewrite sentence once (One-time Rewriting) with Beam Search to select Top-1
        
        Args:
            text: Original sentence to rewrite
            span_texts: List of toxic words/phrases to fix (from Tier 2) - not used directly, only for logging
            spans_indices: List of spans in format [[start, end], ...] (character indices) from Tier 2
            
        Returns:
            Rewritten sentence (Top-1 from beam search)
        """
        try:
            # Use BartphoSpanRewriter from baseline
            # Method rewrite() takes (text, spans_indices)
            if spans_indices:
                # Convert from List[Tuple[int, int]] to List[List[int]]
                spans_indices_list = [[start, end] for start, end in spans_indices]
                rewritten_text = self.rewriter.rewrite(text, spans_indices_list)
            else:
                # Fallback if no spans_indices - use empty list
                rewritten_text = self.rewriter.rewrite(text, [])
            
            # Clean output to remove remaining prompt text
            rewritten_text = self._clean_prediction(rewritten_text, text)
            
            # Ensure we have a result
            if not rewritten_text or not rewritten_text.strip():
                logger.warning("Rewrite result is empty, returning original sentence")
                rewritten_text = text
            
            logger.debug(f"Rewritten sentence: '{text[:50]}...' -> '{rewritten_text[:50]}...'")
            return rewritten_text.strip()
            
        except Exception as e:
            logger.warning(f"Error rewriting sentence: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original sentence if error occurs
            return text
    
    def _clean_prediction(self, prediction: str, original_text: str) -> str:
        """
        Clean prediction to remove remaining prompt text
        
        Args:
            prediction: Text prediction from model
            original_text: Original sentence for comparison
            
        Returns:
            Cleaned text
        """
        if not prediction:
            return prediction
        
        # Use PredictionCleaner from baseline if available
        if self.cleaner:
            prediction = self.cleaner.clean(prediction)
        
        # Remove remaining prompt patterns
        # Pattern 1: "Từ từ" at beginning of sentence (Vietnamese phrase that may leak from prompt)
        prediction = re.sub(r'^Từ từ\s+', '', prediction, flags=re.IGNORECASE)
        
        # Pattern 2: "Câu gốc: ..." or "Câu gốc:... " at the end (original sentence leak)
        prediction = re.sub(r'\s*Câu gốc\s*:\s*.*$', '', prediction, flags=re.IGNORECASE)
        
        # Pattern 3: "Từ cần sửa: ..." or "Từ cần sửa:... " at the end (words to fix leak)
        prediction = re.sub(r'\s*Từ cần sửa\s*:\s*.*$', '', prediction, flags=re.IGNORECASE)
        
        # Pattern 4: "giữ nguyên lập trường và thông tin quan trọng" (may still leak from prompt)
        prediction = re.sub(r'\s*,\s*giữ nguyên lập trường và thông tin quan trọng\.?\s*$', '', prediction, flags=re.IGNORECASE)
        prediction = re.sub(r'\s*giữ nguyên lập trường và thông tin quan trọng\.?\s*$', '', prediction, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        prediction = re.sub(r'\s+', ' ', prediction).strip()
        
        return prediction

