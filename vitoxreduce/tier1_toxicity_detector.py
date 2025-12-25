#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tier 1: Toxicity Detector
Classifies input sentences as safe or unsafe
Uses PhoBERT or viBERT4News
"""

import os
import logging
from typing import Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class ToxicityDetector:
    """Tier 1: Toxicity Detector - Gatekeeper to determine if sentence needs processing"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to PhoBERT classifier model. If None, will raise error
            device: Device to run model on (cuda/cpu)
            threshold: Threshold for unsafe classification (default: 0.5)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PhoBERT classifier model not found at {model_path}. "
                f"Please specify a valid model_path."
            )
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model"""
        logger.info(f"Loading Toxicity Detector from {self.model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Toxicity Detector loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect toxicity of sentence
        
        Args:
            text: Sentence to check
            
        Returns:
            (label, probability): 
                - label: "safe" or "unsafe"
                - probability: Unsafe probability (0.0-1.0)
        """
        if not text or not text.strip():
            return "safe", 0.0
        
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                # Class 1 = unsafe
                unsafe_prob = probs[0][1].item()
            
            label = "unsafe" if unsafe_prob >= self.threshold else "safe"
            return label, float(unsafe_prob)
            
        except Exception as e:
            logger.warning(f"Error detecting toxicity for text '{text[:50]}...': {e}")
            # Default to safe if error occurs
            return "safe", 0.0
    
    def is_safe(self, text: str) -> bool:
        """
        Quickly check if sentence is safe
        
        Returns:
            True if safe, False if unsafe
        """
        label, _ = self.detect(text)
        return label == "safe"
    
    def is_unsafe(self, text: str) -> bool:
        """
        Quickly check if sentence is toxic
        
        Returns:
            True if unsafe, False if safe
        """
        label, _ = self.detect(text)
        return label == "unsafe"

