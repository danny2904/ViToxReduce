#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for PhoBERT Span Locator
Uses trained model to detect toxic spans in text
"""

import os
import logging
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)

# Label mapping
ID_TO_LABEL = {0: "O", 1: "B", 2: "I"}


class SpanLocatorInference:
    """Class for inference with PhoBERT Span Locator model"""
    
    # Class-level flag để chỉ log warning một lần
    _offset_mapping_warning_logged = False
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256
    ):
        """
        Args:
            model_path: Path to trained model
            device: Device to run model on (cuda/cpu)
            max_length: Maximum sequence length
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_path = model_path  # Save model_path for later access
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading Span Locator model from {model_path}...")
        try:
            # Thử load fast tokenizer trước
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                logger.info("Fast tokenizer loaded successfully")
            except Exception:
                logger.warning("Could not load fast tokenizer, using slow tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Span Locator model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Detect toxic spans in text
        
        Args:
            text: Sentence to detect spans in
            
        Returns:
            List of (start_char, end_char) character indices
        """
        if not text or not text.strip():
            return []
        
        try:
            # Tokenize với offset mapping để biết vị trí character
            try:
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_offsets_mapping=True,
                    return_tensors="pt"
                ).to(self.device)
                # Chuyển đổi numpy array thành list of tuples
                offset_mapping_np = encoding["offset_mapping"][0].cpu().numpy()
                offset_mapping = [(int(start), int(end)) for start, end in offset_mapping_np]
            except (NotImplementedError, TypeError) as e:
                # Tokenizer không hỗ trợ offset mapping
                # Chỉ log warning một lần để tránh spam
                if not SpanLocatorInference._offset_mapping_warning_logged:
                    logger.warning(
                        f"Tokenizer does not support offset mapping: {e}. "
                        f"Using manual computation method to return character spans. "
                        f"(This warning will only be shown once)"
                    )
                    SpanLocatorInference._offset_mapping_warning_logged = True
                
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Tính offset mapping thủ công
                offset_mapping = self._compute_offset_mapping_manual(text, encoding["input_ids"][0])
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in encoding.items() if k != "offset_mapping"})
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # Chuyển đổi token-level predictions thành character-level spans
            spans = self._tokens_to_char_spans(
                text, predictions, offset_mapping
            )
            
            return spans
            
        except Exception as e:
            logger.warning(f"Error detecting spans for text '{text[:50]}...': {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def _compute_offset_mapping_manual(self, text: str, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Manually compute offset mapping when tokenizer doesn't support it
        
        Args:
            text: Original text
            input_ids: Token IDs from tokenizer
            
        Returns:
            List of (char_start, char_end) tuples
        """
        offset_mapping = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        char_pos = 0
        
        for token in tokens:
            # Xử lý special tokens
            if token == self.tokenizer.pad_token or token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
                offset_mapping.append((0, 0))
                continue
            
            # Decode token về text gốc (loại bỏ special prefixes)
            # PhoBERT sử dụng BPE, token có thể có prefix như "▁" hoặc không
            decoded_token = self.tokenizer.convert_tokens_to_string([token]).strip()
            
            if not decoded_token:
                offset_mapping.append((0, 0))
                continue
            
            # Tìm vị trí của decoded token trong text từ vị trí hiện tại
            if char_pos < len(text):
                # Tìm từ vị trí char_pos trở đi
                search_text = text[char_pos:]
                found_pos = search_text.find(decoded_token)
                
                if found_pos != -1:
                    # Tìm thấy, cập nhật vị trí
                    actual_pos = char_pos + found_pos
                    offset_mapping.append((actual_pos, actual_pos + len(decoded_token)))
                    char_pos = actual_pos + len(decoded_token)
                else:
                    # Không tìm thấy chính xác, thử tìm không phân biệt hoa thường
                    search_text_lower = search_text.lower()
                    found_pos = search_text_lower.find(decoded_token.lower())
                    if found_pos != -1:
                        actual_pos = char_pos + found_pos
                        offset_mapping.append((actual_pos, actual_pos + len(decoded_token)))
                        char_pos = actual_pos + len(decoded_token)
                    else:
                        # Vẫn không tìm thấy, ước tính dựa trên độ dài
                        offset_mapping.append((char_pos, min(char_pos + len(decoded_token), len(text))))
                        char_pos = min(char_pos + len(decoded_token), len(text))
            else:
                # Đã vượt quá độ dài text
                offset_mapping.append((0, 0))
        
        return offset_mapping
    
    def _tokens_to_char_spans(
        self,
        text: str,
        token_predictions: List[int],
        offset_mapping: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Convert token-level predictions (BIO) to character-level spans
        
        Args:
            text: Original text
            token_predictions: List of label IDs (0=O, 1=B, 2=I)
            offset_mapping: List of (char_start, char_end) for each token
            
        Returns:
            List of (start_char, end_char) spans
        """
        spans = []
        current_span_start = None
        
        for token_idx, (label_id, (char_start, char_end)) in enumerate(zip(token_predictions, offset_mapping)):
            # Bỏ qua special tokens và padding
            if char_start == 0 and char_end == 0:
                continue
            
            label = ID_TO_LABEL.get(label_id, "O")
            
            if label == "B":
                # Bắt đầu span mới
                # Kết thúc span cũ nếu có
                if current_span_start is not None:
                    spans.append((current_span_start, char_start))
                current_span_start = char_start
            elif label == "I":
                # Tiếp tục span hiện tại
                if current_span_start is not None:
                    # Cập nhật end position
                    pass  # Sẽ cập nhật ở cuối
                else:
                    # Nếu không có B trước đó, coi như B
                    current_span_start = char_start
            else:  # label == "O"
                # Kết thúc span hiện tại nếu có
                if current_span_start is not None:
                    spans.append((current_span_start, char_start))
                    current_span_start = None
        
        # Kết thúc span cuối cùng nếu có
        if current_span_start is not None:
            spans.append((current_span_start, len(text)))
        
        # Merge overlapping spans
        return self._merge_overlapping_spans(spans)
    
    def _merge_overlapping_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping spans"""
        if not spans:
            return []
        
        # Sort theo start position
        sorted_spans = sorted(spans, key=lambda x: x[0])
        merged = []
        cur_start, cur_end = sorted_spans[0]
        
        for start, end in sorted_spans[1:]:
            if start <= cur_end:
                # Overlap, merge
                cur_end = max(cur_end, end)
            else:
                # Không overlap, lưu span hiện tại và bắt đầu span mới
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        
        merged.append((cur_start, cur_end))
        return merged

