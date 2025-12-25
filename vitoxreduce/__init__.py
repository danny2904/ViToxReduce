#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline ViToxReduce - Hệ thống giảm độc hại văn bản tiếng Việt
Kiến trúc 3 tầng Streamlined: Check → Tag → Rewrite
"""

from .pipeline import ViToxReducePipeline
from .tier1_toxicity_detector import ToxicityDetector
from .tier2_span_locator import SpanTagger
from .tier3_rewrite_generator import ContextualRewriter

__all__ = [
    'ViToxReducePipeline',
    'ToxicityDetector',
    'SpanTagger',
    'ContextualRewriter',
]

__version__ = '2.0.0'

