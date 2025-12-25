#!/usr/bin/env python3
"""
Bartpho span-guided rewriter: uses fine-tuned model to rewrite toxic sentences based on spans
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger("bartpho_span_baseline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

SPECIAL_REPEAT_PATTERN = re.compile(r'([!?.=~@#$%^&*()_+:;"\'<>,])\1{2,}')
MULTISPACE_PATTERN = re.compile(r"\s+")

TOX_TAG_OPEN = "<tox>"
TOX_TAG_CLOSE = "</tox>"
SOFT_DETOX_INSTRUCTION = "Viết lại câu sau ít xúc phạm hơn, giữ nguyên lập trường và thông tin quan trọng."
SPAN_GUIDELINES = [
    "- Thay toàn bộ nội dung nằm trong <tox>...</tox> bằng diễn đạt mềm hơn.",
    "- Không giữ nguyên bất kỳ từ/cụm từ nào bên trong tag <tox>.",
    "- Giữ nguyên mọi phần không được đánh dấu để tránh thay đổi thông tin.",
]


class PredictionCleaner:
    """Light cleaning of prediction."""

    def __init__(self, enable: bool = True):
        self.enable = enable

    @staticmethod
    def _sentence_case(text: str) -> str:
        if not text:
            return text
        lowered = text.lower()
        for idx, ch in enumerate(lowered):
            if ch.isalpha():
                return lowered[:idx] + ch.upper() + lowered[idx + 1 :]
        return lowered

    def clean(self, text: str) -> str:
        if not self.enable or not text:
            return text
        cleaned = text.strip()
        cleaned = SPECIAL_REPEAT_PATTERN.sub(r"\1\1", cleaned)
        if cleaned.isupper() and len(cleaned) > 6:
            cleaned = self._sentence_case(cleaned)
        cleaned = MULTISPACE_PATTERN.sub(" ", cleaned)
        return cleaned.strip()


def normalize_rewrite_field(rewrites: Any) -> str:
    if not rewrites:
        return ""
    if isinstance(rewrites, str):
        return rewrites.strip()
    if isinstance(rewrites, Sequence):
        for candidate in rewrites:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def build_span_guided_prompt(marked_comment: str, span_texts: List[str]) -> str:
    prompt_lines = [SOFT_DETOX_INSTRUCTION] + SPAN_GUIDELINES
    prompt_lines.append(f"Câu đã đánh dấu: {marked_comment}")
    span_list = [s.strip() for s in span_texts if s and s.strip()]
    if span_list:
        prompt_lines.append(f"Các cụm độc hại cần thay: {'; '.join(span_list)}")
    prompt_lines.append("Ví dụ: <tox>nước mắt cá sấu</tox> → giả vờ thương xót.")
    return "\n".join(prompt_lines)


def mark_toxic_spans(text: str, spans_indices: List[List[int]]) -> Tuple[str, List[str]]:
    if not spans_indices:
        return text, []
    spans_sorted = sorted(spans_indices, key=lambda x: x[0])
    merged: List[List[int]] = []
    cur_s, cur_e = spans_sorted[0]
    for s, e in spans_sorted[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    merged.append([cur_s, cur_e])

    pieces: List[str] = []
    span_texts: List[str] = []
    last = 0
    for start, end in merged:
        start = max(0, min(len(text), start))
        end = max(start, min(len(text), end))
        if start > last:
            pieces.append(text[last:start])
        span_txt = text[start:end]
        span_texts.append(span_txt)
        pieces.append(f"{TOX_TAG_OPEN}{span_txt}{TOX_TAG_CLOSE}")
        last = end
    if last < len(text):
        pieces.append(text[last:])
    return "".join(pieces), span_texts


class BartphoSpanRewriter:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 256,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        top_p: float = 0.9,
        temperature: float = 0.9,
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        self.num_return_sequences = max(1, num_return_sequences)
        self.do_sample = do_sample or self.num_return_sequences > 1
        self.top_p = top_p
        self.temperature = temperature
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BARTpho model not found at {model_path}")
        logger.info("Loading BARTpho span model from %s...", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_input(self, text: str, spans_indices: List[List[int]]) -> Tuple[str, List[str]]:
        marked_comment, span_texts = mark_toxic_spans(text, spans_indices)
        if span_texts:
            input_text = build_span_guided_prompt(marked_comment, span_texts)
        else:
            input_text = f"{SOFT_DETOX_INSTRUCTION}\nCâu gốc: {text}"
        return input_text, span_texts

    def generate_candidates(self, text: str, spans_indices: List[List[int]]) -> List[str]:
        input_text, _ = self._build_input(text, spans_indices)
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_return_sequences,
                    early_stopping=True,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=3.5,
                    no_repeat_ngram_size=2,
                    length_penalty=1.2,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                )
            return [
                (self.tokenizer.decode(output, skip_special_tokens=True).strip() or text)
                for output in outputs
            ]
        except Exception as exc:  # pragma: no cover
            logger.warning("Error generating rewrite with BARTpho: %s", exc)
            return [text]

    def rewrite(self, text: str, spans_indices: List[List[int]]) -> str:
        return self.generate_candidates(text, spans_indices)[0]

