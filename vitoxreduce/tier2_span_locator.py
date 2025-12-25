#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 2: Span Tagger
Uses PhoBERT Token Classification (BIO Tagging) to identify toxic phrase locations
"""

import os
import json
import logging
import torch
from typing import List, Tuple, Optional, Dict

from .span_locator_inference import SpanLocatorInference

logger = logging.getLogger(__name__)


class SpanTagger:
    """Tier 2: Span Tagger - PhoBERT Token Classification (BIO Tagging)"""
    
    def __init__(
        self,
        span_locator_model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
        span_dictionary_paths: Optional[List[str]] = None
    ):
        """
        Args:
            span_locator_model_path: Path to PhoBERT Token Classification model (required)
            device: Device to run model on (cuda/cpu)
            max_length: Maximum sequence length
            span_dictionary_paths: List of paths to JSONL files to create span dictionary (optional)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.span_locator_model = None
        self.span_dictionary: Dict[str, List[Tuple[int, int]]] = {}  # text -> spans_indices (legacy format)
        self.span_texts_set: set = set()  # Set of toxic span texts (new format)
        
        if span_locator_model_path is None or not os.path.exists(span_locator_model_path):
            raise FileNotFoundError(
                f"PhoBERT Span Locator model not found at {span_locator_model_path}. "
                f"Please specify a valid span_locator_model_path."
            )
        
        # Load span dictionary if available
        if span_dictionary_paths:
            self._load_span_dictionary(span_dictionary_paths)
        else:
            logger.info("No span dictionary provided, will use model only")
        
        # Load Span Locator model
        self.span_locator_model_path = span_locator_model_path
        self._load_span_locator_model(span_locator_model_path)
    
    def _load_span_dictionary(self, file_paths: List[str]):
        """
        Load span dictionary from JSONL files (extracted or train/val)
        
        Args:
            file_paths: List of paths to JSONL files
        """
        logger.info(f"Loading span dictionary from {len(file_paths)} file(s)...")
        total_loaded = 0
        total_spans_loaded = 0
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            
                            # Format mới: {"span": "...", "length": ...}
                            if 'span' in item:
                                span_text = item.get('span', '').strip()
                                # Loại bỏ dấu ngoặc kép ở đầu và cuối nếu có
                                if span_text.startswith('"') and span_text.endswith('"'):
                                    span_text = span_text[1:-1].strip()
                                if span_text:
                                    self.span_texts_set.add(span_text)
                                    total_spans_loaded += 1
                            
                            # Format cũ: {"comment": "...", "unsafe_spans_indices": [...]}
                            elif 'comment' in item:
                                comment = item.get('comment', '').strip()
                                unsafe_spans_indices = item.get('unsafe_spans_indices', [])
                                
                                if not comment:
                                    continue
                                
                                if unsafe_spans_indices:
                                    # Chuyển đổi từ list of lists sang list of tuples
                                    spans_indices = []
                                    for span in unsafe_spans_indices:
                                        if isinstance(span, (list, tuple)) and len(span) == 2:
                                            start, end = int(span[0]), int(span[1])
                                            # Validate span indices
                                            if 0 <= start < end <= len(comment):
                                                spans_indices.append((start, end))
                                    
                                    if spans_indices:
                                        # Lưu vào từ điển (key là comment text)
                                        # Nếu đã có, giữ nguyên (không ghi đè)
                                        if comment not in self.span_dictionary:
                                            self.span_dictionary[comment] = spans_indices
                                            total_loaded += 1
                        except json.JSONDecodeError as e:
                            logger.debug(f"Skipping invalid JSON line {line_num} in {file_path}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Error loading dictionary from {file_path}: {e}")
                continue
        
        if total_spans_loaded > 0:
            logger.info(f"Loaded {total_spans_loaded} spans into span dictionary (new format)")
            # Log một số spans mẫu để debug
            sample_spans = sorted(list(self.span_texts_set))[:10]
            logger.debug(f"Sample spans in dictionary: {sample_spans}")
        if total_loaded > 0:
            logger.info(f"Loaded {total_loaded} entries into span dictionary (old format, total {len(self.span_dictionary)} unique texts)")
    
    def _load_span_locator_model(self, model_path: str):
        """Load PhoBERT Span Locator model"""
        try:
            self.span_locator_model = SpanLocatorInference(
                model_path=model_path,
                device=self.device,
                max_length=self.max_length
            )
            logger.info("Span Tagger model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Span Locator model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def locate_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Identify toxic spans in text
        Prioritize using span dictionary (from train/val) first, only call model if not found
        
        Args:
            text: Sentence to find spans in
            
        Returns:
            List of (start, end) character indices
        """
        logger.debug(f"[Tier2] Finding spans for text: '{text[:100]}...'")
        
        # Bước 1: Kiểm tra từ điển span format cũ (text -> spans_indices)
        if text in self.span_dictionary:
            spans = self.span_dictionary[text]
            logger.debug(f"[Tier2] Found spans in dictionary (old format) for text '{text[:50]}...': {len(spans)} spans")
            return spans
        
        # Bước 2: Kiểm tra từ điển span format mới (tìm kiếm substring)
        if self.span_texts_set:
            found_spans = []
            text_lower = text.lower()  # Tìm kiếm không phân biệt hoa thường
            
            # Sắp xếp các span theo độ dài giảm dần để ưu tiên span dài hơn
            sorted_spans = sorted(self.span_texts_set, key=len, reverse=True)
            
            logger.debug(f"[Tier2] Searching {len(sorted_spans)} spans in text (dictionary new format)")
            
            for span_text in sorted_spans:
                span_text_clean = span_text.strip()
                if not span_text_clean:  # Bỏ qua span rỗng
                    continue
                
                span_lower = span_text_clean.lower()
                
                # Tìm tất cả các vị trí xuất hiện của span trong text
                start = 0
                while True:
                    pos = text_lower.find(span_lower, start)
                    if pos == -1:
                        break
                    # Tính end dựa trên độ dài của span_text_clean (đã strip)
                    end = pos + len(span_text_clean)
                    # Validate span indices
                    if 0 <= pos < end <= len(text):
                        # Kiểm tra xem span này có bị overlap với span đã tìm thấy không
                        overlap = False
                        for existing_start, existing_end in found_spans:
                            # Overlap nếu: không phải là (end <= existing_start) và không phải là (pos >= existing_end)
                            if not (end <= existing_start or pos >= existing_end):
                                overlap = True
                                logger.debug(f"Span '{span_text_clean}' at ({pos}, {end}) overlaps with span at ({existing_start}, {existing_end})")
                                break
                        if not overlap:
                            found_spans.append((pos, end))
                            logger.debug(f"Found span '{span_text_clean}' at position ({pos}, {end}) in text")
                    start = pos + 1
            
            if found_spans:
                # Sắp xếp theo vị trí bắt đầu
                found_spans = sorted(found_spans)
                logger.debug(f"[Tier2] ✓ Found {len(found_spans)} spans in dictionary (new format) for text '{text[:50]}...': {found_spans}")
                return found_spans
            else:
                logger.debug(f"[Tier2] No spans found in dictionary (new format) for text '{text[:50]}...'")
        
        # Bước 3: Nếu không có trong từ điển, gọi model
        logger.debug(f"[Tier2] Not found in dictionary, calling model to predict spans for text '{text[:50]}...'")
        if self.span_locator_model is None:
            logger.warning("[Tier2] Span Locator model not loaded and not found in dictionary")
            return []
        
        try:
            spans = self.span_locator_model.predict_spans(text)
            logger.debug(f"[Tier2] Model predicted {len(spans)} spans for text '{text[:50]}...': {spans}")
            return spans
        except Exception as e:
            logger.warning(f"[Tier2] Error detecting spans for text '{text[:50]}...': {e}")
            return []
    
    def get_span_texts(self, text: str, spans_indices: List[Tuple[int, int]]) -> List[str]:
        """
        Get text of spans from indices
        
        Args:
            text: Original sentence
            spans_indices: List of (start, end) character indices
            
        Returns:
            List of span texts
        """
        span_texts = []
        for start, end in spans_indices:
            if 0 <= start < end <= len(text):
                span_texts.append(text[start:end])
        return span_texts

