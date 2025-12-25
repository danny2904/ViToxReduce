#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main ViToxReduce Pipeline - Streamlined 3-Tier Architecture
Tier 1: Safety Checker (PhoBERT Sequence Classification)
Tier 2: Span Tagger (PhoBERT Token Classification / BIO Tagging)
Tier 3: Contextual Rewriter (BARTpho Seq2Seq - Span-guided)
"""

import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
import torch
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

from .tier1_toxicity_detector import ToxicityDetector
from .tier2_span_locator import SpanTagger
from .tier3_rewrite_generator import ContextualRewriter

logger = logging.getLogger(__name__)


class ViToxReducePipeline:
    """ViToxReduce Pipeline with Streamlined 3-Tier Architecture"""
    
    def __init__(
        self,
        # Tier 1 config
        toxicity_detector_model_path: Optional[str] = None,
        toxicity_threshold: float = 0.5,
        
        # Tier 2 config
        span_locator_model_path: Optional[str] = None,
        
        # Tier 3 config
        rewriter_model_path: Optional[str] = None,
        num_beams: int = 5,
        
        # General config
        device: Optional[str] = None,
    ):
        """
        Initialize pipeline with configs for each tier
        
        Args:
            toxicity_detector_model_path: Path to PhoBERT classifier for Tier 1
            toxicity_threshold: Threshold for unsafe classification
            span_locator_model_path: Path to PhoBERT Token Classification model for Tier 2
            rewriter_model_path: Path to BARTpho rewriter model for Tier 3 (required)
            num_beams: Number of beams for beam search in Tier 3
            device: Device to run models on
        """
        if rewriter_model_path is None:
            raise ValueError("rewriter_model_path is required and cannot be None")
        
        # Determine device
        if device:
            self.device = device.lower()
            if self.device not in ["cuda", "cpu"]:
                logger.warning(f"Device '{device}' is invalid, using 'cpu'")
                self.device = "cpu"
        else:
            # Auto-select device
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        # Log device information
        logger.info("=" * 60)
        logger.info("INITIALIZING VITOXREDUCE PIPELINE (3 TIERS)")
        logger.info("=" * 60)
        logger.info(f"Device in use: {self.device.upper()}")
        if self.device == "cuda":
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                logger.warning("CUDA not available but device is set to 'cuda', switching to CPU")
                self.device = "cpu"
        else:
            logger.info("Using CPU (may be slower than GPU)")
        
        # Tầng 1: Safety Checker
        logger.info("Initializing Tier 1: Safety Checker (PhoBERT Sequence Classification)...")
        self.tier1 = ToxicityDetector(
            model_path=toxicity_detector_model_path,
            device=self.device,
            threshold=toxicity_threshold
        )
        
        # Tier 2: Span Tagger
        logger.info("Initializing Tier 2: Span Tagger (PhoBERT Token Classification)...")
        self.tier2 = SpanTagger(
            span_locator_model_path=span_locator_model_path,
            device=self.device
        )
        
        # Tier 3: Contextual Rewriter
        logger.info("Initializing Tier 3: Contextual Rewriter (BARTpho Seq2Seq)...")
        self.tier3 = ContextualRewriter(
            model_path=rewriter_model_path,
            device=self.device,
            num_beams=num_beams
        )
        
        logger.info("=" * 60)
        logger.info("PIPELINE READY!")
        logger.info("=" * 60)
    
    def process(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a sentence through the 3-tier pipeline: Check → Tag → Rewrite
        
        Args:
            text: Sentence to process
            verbose: Print detailed processing information
            
        Returns:
            Dict containing result with keys:
                - original: Original sentence
                - is_safe: True if safe, False if unsafe
                - toxicity_score: Toxicity score of original sentence (0.0-1.0)
                - spans: List of found spans (character indices)
                - span_texts: List of span texts
                - rewritten: Rewritten sentence (or original if safe)
                - rewritten_is_safe: True if rewritten sentence is safe, False if still unsafe, None if not checked
                - rewritten_toxicity_score: Toxicity score of rewritten sentence (0.0-1.0), None if not checked
                - processing_time: Processing time (seconds)
        """
        result = {
            'original': text,
            'is_safe': False,
            'toxicity_score': 0.0,
            'spans': [],
            'span_texts': [],
            'rewritten': text,
            'rewritten_is_safe': None,  # None if not checked yet, True/False after check
            'rewritten_toxicity_score': None,  # None if not checked yet, score after check
            'processing_time': 0.0,
        }
        
        start_time = datetime.now()
        
        # Tầng 1: Safety Checker
        if verbose:
            logger.info(f"\n[Tier 1] Safety check: '{text[:50]}...'")
        label, toxicity_score = self.tier1.detect(text)
        result['is_safe'] = (label == "safe")
        result['toxicity_score'] = toxicity_score
        
        if result['is_safe']:
            # Safe sentence, keep unchanged and terminate
            # NOTE: Safe sentences SKIP tier 2 and tier 3, do not run span locator and rewriter
            # spans and span_texts will remain [] (empty)
            if verbose:
                logger.info(f"[Tier 1] Sentence is safe (toxicity={toxicity_score:.3f}), keeping unchanged")
            result['rewritten'] = text
            result['rewritten_is_safe'] = True  # Original is safe, so rewritten is also safe
            result['rewritten_toxicity_score'] = toxicity_score
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            return result
        
        # Unsafe sentence, continue processing through tier 2 and tier 3
        if verbose:
            logger.info(f"[Tier 1] Sentence is unsafe (toxicity={toxicity_score:.3f}), continuing processing...")
        
        # Tier 2: Span Tagger (ONLY runs when unsafe)
        if verbose:
            logger.info(f"[Tier 2] Locating toxic spans...")
        spans_indices = self.tier2.locate_spans(text)
        result['spans'] = spans_indices  # Save character indices of spans
        
        # Get text of spans
        span_texts = self.tier2.get_span_texts(text, spans_indices)
        result['span_texts'] = span_texts  # Save span texts
        
        if verbose:
            logger.info(f"[Tier 2] Found {len(spans_indices)} spans: {span_texts}")
        
        # Tier 3: Contextual Rewriter
        if verbose:
            logger.info(f"[Tier 3] Rewriting sentence (One-time Rewriting with Beam Search)...")
        # Pass spans_indices to tier3 to use BartphoSpanRewriter from baseline
        rewritten_text = self.tier3.rewrite(text, span_texts=span_texts, spans_indices=spans_indices)
        result['rewritten'] = rewritten_text
        
        if verbose:
            logger.info(f"[Tier 3] Result: '{text[:50]}...' -> '{rewritten_text[:50]}...'")
        
        # Re-check rewritten sentence with Tier 1 to evaluate effectiveness
        if verbose:
            logger.info(f"[Tier 1 - Post-check] Re-checking rewritten sentence...")
        rewritten_label, rewritten_toxicity_score = self.tier1.detect(rewritten_text)
        result['rewritten_is_safe'] = (rewritten_label == "safe")
        result['rewritten_toxicity_score'] = rewritten_toxicity_score
        
        if verbose:
            status = "SAFE" if result['rewritten_is_safe'] else "STILL TOXIC"
            logger.info(f"[Tier 1 - Post-check] Rewritten sentence: {status} (toxicity={rewritten_toxicity_score:.3f})")
            logger.info(f"[Tier 1 - Post-check] Toxicity reduction: {result['toxicity_score']:.3f} -> {rewritten_toxicity_score:.3f} (Δ={result['toxicity_score'] - rewritten_toxicity_score:.3f})")
        
        result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def process_batch(self, texts: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Process a batch of sentences
        
        Args:
            texts: List of sentences to process
            verbose: Print detailed processing information
            
        Returns:
            List of results
        """
        results = []
        
        # Use tqdm to display progress bar
        if TQDM_AVAILABLE:
            text_iterator = tqdm(
                enumerate(texts),
                total=len(texts),
                desc="Processing",
                unit="sentences",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            text_iterator = enumerate(texts)
            logger.warning("tqdm not available, progress bar will not be displayed")
        
        for i, text in text_iterator:
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing sentence {i+1}/{len(texts)}")
                logger.info(f"{'='*60}")
            result = self.process(text, verbose=verbose)
            results.append(result)
            
            # Update progress bar description with detailed information
            if TQDM_AVAILABLE and not verbose:
                text_iterator.set_postfix({
                    'safe': sum(1 for r in results if r.get('is_safe', False)),
                    'unsafe': sum(1 for r in results if not r.get('is_safe', True))
                })
        
        return results


if __name__ == "__main__":
    # Setup logging before using logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Get logger again after logging is configured
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("VITOXREDUCE PIPELINE - MODULE (3 TIERS)")
    print("=" * 60)
    print("This file contains the ViToxReducePipeline class.")
    print("To run the pipeline, please use:")
    print("  python run_pipeline.py --input <input> --rewriter_model <model_path>")
    print("")
    print("Or import and use in code:")
    print("  from vitoxreduce_pipeline import ViToxReducePipeline")
    print("  pipeline = ViToxReducePipeline(rewriter_model_path='...')")
    print("  result = pipeline.process('text to process', verbose=True)")
    print("=" * 60)

