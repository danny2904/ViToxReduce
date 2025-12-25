#!/usr/bin/env python3
"""
Common evaluation utilities for all baseline scripts:
- Vietnamese tokenization using PyVi
- Metrics: BLEU, SIM (SBERT), Fluency (GPT-2 Vietnamese PPL based), STA (Toxicity Drop using PhoBERT), J-score (Joint score)
"""

from __future__ import annotations

import logging
import math
import traceback
from typing import Any, Dict, List, Optional, Sequence

import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# Logger will propagate to root logger so main scripts can capture
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Log INFO and above
logger.propagate = True  # Propagate to root logger

# Helper function to get logger - prioritize root logger if it has a handler
def get_eval_logger():
    """Get logger, prioritize root logger if it has a handler"""
    root_logger = logging.getLogger()
    # If root logger has handler, use it
    if root_logger.handlers:
        return root_logger
    # Otherwise, use this module's logger
    return logger

# --- Optional Imports ---
try:
    from pyvi import ViTokenizer  # type: ignore
    _VI_TOKENIZER_AVAILABLE = True
except ImportError:
    ViTokenizer = None
    _VI_TOKENIZER_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
    _BLEU_SMOOTH = SmoothingFunction().method1
except ImportError:
    BLEU_AVAILABLE = False
    _BLEU_SMOOTH = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_MODEL_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_MODEL_AVAILABLE = False

# --- Global Models ---
_sentence_model: Optional["SentenceTransformer"] = None
_fluency_tokenizer: Optional[AutoTokenizer] = None
_fluency_model: Optional[AutoModelForCausalLM] = None
# Use GPT-2 Vietnamese model to calculate PPL
_FLUENCY_MODEL_NAMES = [
    "NlpHUST/gpt2-vietnamese",
]

# PhoBERT classifier for toxicity calculation
_phobert_model: Optional[AutoModelForSequenceClassification] = None
_phobert_tokenizer: Optional[AutoTokenizer] = None
_phobert_device: str = "cuda" if torch.cuda.is_available() else "cpu"

# --- Tokenization ---
def tokenize_vietnamese(text: str) -> List[str]:
    """Tokenize Vietnamese text using PyVi when available."""
    if _VI_TOKENIZER_AVAILABLE and ViTokenizer is not None:
        try:
            return ViTokenizer.tokenize(text).split()
        except Exception as exc:
            logger.warning("ViTokenizer error, fallback to split: %s", exc)
    return text.split()

# --- Metric: BLEU ---
def calculate_bleu_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    if not BLEU_AVAILABLE or not _BLEU_SMOOTH:
        return {"bleu": 0.0}

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_vietnamese(pred.lower())
        ref_tokens = tokenize_vietnamese(ref.lower())
        if not ref_tokens:
            continue
        try:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=_BLEU_SMOOTH)
            scores.append(bleu)
        except ZeroDivisionError:
            continue

    if not scores:
        return {"bleu": 0.0}
    return {"bleu": float(np.mean(scores))}

# --- Metric: SIM (Semantic Similarity) ---
def get_sentence_model() -> Optional["SentenceTransformer"]:
    global _sentence_model
    if _sentence_model is None and SENTENCE_MODEL_AVAILABLE:
        try:
            _sentence_model = SentenceTransformer("keepitreal/vietnamese-sbert")
        except Exception:
            try:
                # Fallback model
                _sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2") 
            except Exception:
                _sentence_model = None
    return _sentence_model

def calculate_sim(predictions: List[str], originals: List[str]) -> Dict[str, float]:
    model = get_sentence_model()
    if model is None:
        return {"sim": 0.0}
    
    try:
        # Encode batch để nhanh hơn
        orig_emb = model.encode(originals, show_progress_bar=False)
        pred_emb = model.encode(predictions, show_progress_bar=False)
        
        sims = []
        for o_vec, p_vec in zip(orig_emb, pred_emb):
            # Cosine similarity trả về ma trận, lấy phần tử [0][0]
            val = cosine_similarity([o_vec], [p_vec])[0][0]
            sims.append(float(val))
            
        return {"sim": float(np.mean(sims))}
    except Exception as exc:
        logger.warning("Error calculating SIM: %s", exc)
        return {"sim": 0.0}

# --- Metric: Fluency (Based on GPT-2 Vietnamese PPL) ---
def get_fluency_resources() -> Optional[tuple[AutoTokenizer, AutoModelForCausalLM]]:
    global _fluency_tokenizer, _fluency_model
    if _fluency_model is None or _fluency_tokenizer is None:
        eval_logger = get_eval_logger()
        msg = "[eval_metrics] Starting to load GPT-2 Vietnamese model..."
        print(msg)
        eval_logger.info(msg)
        
        # Đảm bảo logger có handler để ghi log
        if not eval_logger.handlers:
            # Thêm console handler nếu chưa có
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            eval_logger.addHandler(handler)
        
        # Thử load từng model theo thứ tự ưu tiên
        last_error = None
        for idx, model_name in enumerate(_FLUENCY_MODEL_NAMES):
            try:
                msg = f"[eval_metrics] Attempting to load GPT-2 Vietnamese model ({idx+1}/{len(_FLUENCY_MODEL_NAMES)}): {model_name}..."
                print(msg)
                eval_logger.info(msg)
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.eval()
                
                # Move to device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                
                msg = f"[eval_metrics] Successfully loaded GPT-2 Vietnamese: {model_name}, device: {device}"
                print(msg)
                eval_logger.info(msg)
                
                _fluency_tokenizer = tokenizer
                _fluency_model = model
                return _fluency_tokenizer, _fluency_model
                
            except Exception as exc:
                last_error = exc
                error_detail = traceback.format_exc()
                msg = f"[eval_metrics] Could not load {model_name}: {exc}"
                print(msg)
                eval_logger.warning(msg)
                eval_logger.debug(f"[eval_metrics] Error details for loading {model_name}:\n{error_detail}")
                # Lưu traceback để log sau
                if not hasattr(get_fluency_resources, '_last_traceback'):
                    get_fluency_resources._last_traceback = error_detail
                continue
        
        # Nếu tất cả đều fail
        eval_logger = get_eval_logger()
        error_msgs = [
            "[eval_metrics] ⚠️ COULD NOT LOAD ANY GPT-2 Vietnamese MODEL!",
            "[eval_metrics] PPL and FL will be 0.0. Please check:",
            "[eval_metrics]   1. Does the model exist on HuggingFace?",
            "[eval_metrics]   2. Is there enough RAM/VRAM?",
            "[eval_metrics]   3. Is there internet connection to download the model?",
        ]
        if last_error:
            error_msgs.append(f"[eval_metrics] Last error: {last_error}")
            if hasattr(get_fluency_resources, '_last_traceback'):
                error_msgs.append(f"[eval_metrics] Traceback error:\n{get_fluency_resources._last_traceback}")
        
        for msg in error_msgs:
            print(msg)
            eval_logger.error(msg)
        return None
    return _fluency_tokenizer, _fluency_model

def calculate_perplexity_and_fluency(texts: List[str]) -> Dict[str, float]:
    """
    Calculate Perplexity (PPL) and convert to Fluency score (0-1).
    Fluency Score = 1 / (1 + 0.05 * PPL)
    """
    # Ensure log is written
    eval_logger = get_eval_logger()
    msg = f"[eval_metrics] Starting to calculate PPL and FL for {len(texts)} sentences..."
    print(msg)
    eval_logger.info(msg)
    
    resources = get_fluency_resources()
    if resources is None:
        msg = "[eval_metrics] ⚠️ GPT-2 Vietnamese model not available, PPL and FL = 0.0"
        print(msg)
        eval_logger.warning(msg)
        # Log thêm thông tin debug
        eval_logger.warning("[eval_metrics] Check: _fluency_model = %s, _fluency_tokenizer = %s", 
                           _fluency_model is not None, _fluency_tokenizer is not None)
        return {"ppl": 0.0, "fl": 0.0}
    
    tokenizer, model = resources
    device = model.device
    
    ppls = []
    fl_scores = []
    error_count = 0
    
    # Tính batch hoặc từng câu (ở đây làm từng câu cho an toàn memory)
    for text in texts:
        if not text.strip():
            ppls.append(0.0)
            fl_scores.append(0.0)
            continue
            
        try:
            # Tokenize text (không cần padding cho single text)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            with torch.no_grad():
                # Tính logits
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Tính loss thủ công để đảm bảo chỉ tính trên non-padding tokens
                # Shift labels và logits để tính next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Tính cross-entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
                # Flatten để tính loss
                shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels_flat = shift_labels.view(-1)
                
                # Tính loss cho từng token
                token_losses = loss_fct(shift_logits_flat, shift_labels_flat)
                
                # Nếu có attention_mask, chỉ tính loss trên non-padding tokens
                if attention_mask is not None:
                    # Shift attention_mask để khớp với shift_labels
                    shift_attention = attention_mask[..., 1:].contiguous().view(-1)
                    # Chỉ lấy loss của các token không phải padding
                    valid_losses = token_losses[shift_attention == 1]
                    if len(valid_losses) > 0:
                        loss = valid_losses.mean()
                    else:
                        # Nếu không có token hợp lệ, dùng giá trị mặc định
                        loss = torch.tensor(10.0, device=device)  # PPL ≈ 22k
                else:
                    # Không có attention_mask, tính trung bình tất cả (trừ padding tokens nếu có)
                    if tokenizer.pad_token_id is not None:
                        valid_losses = token_losses[shift_labels_flat != tokenizer.pad_token_id]
                        if len(valid_losses) > 0:
                            loss = valid_losses.mean()
                        else:
                            loss = torch.tensor(10.0, device=device)
                    else:
                        loss = token_losses.mean()
            
            ppl = math.exp(loss.item())
            
            # Xử lý PPL vô cùng lớn (tránh lỗi)
            if math.isnan(ppl) or math.isinf(ppl):
                ppl = 1e6
                
            ppls.append(ppl)
            
            # Quy đổi PPL -> Fluency Score (0-1)
            # Công thức soft: score = 1 / (1 + 0.05 * ppl)
            # Ví dụ: PPL=20 -> fl = 1/(1+1) = 0.5. PPL=10 -> fl=0.66
            fl = 1.0 / (1.0 + 0.05 * ppl)
            fl_scores.append(fl)
            
        except Exception as exc:
            error_count += 1
            ppls.append(0.0)
            fl_scores.append(0.0)
            if error_count <= 3:  # Only log first 3 errors to avoid spam
                eval_logger = get_eval_logger()
                error_detail = traceback.format_exc()
                eval_logger.warning(f"[eval_metrics] Error calculating PPL for text '{text[:50]}...': {exc}")
                eval_logger.debug(f"[eval_metrics] Error details:\n{error_detail}")

    eval_logger = get_eval_logger()
    if error_count > 0:
        msg = f"[eval_metrics] {error_count}/{len(texts)} sentences failed when calculating PPL"
        print(msg)
        eval_logger.warning(msg)
    
    avg_ppl = float(np.mean(ppls)) if ppls else 0.0
    avg_fl = float(np.mean(fl_scores)) if fl_scores else 0.0
    
    if avg_ppl == 0.0 and avg_fl == 0.0 and len(texts) > 0:
        msg = "[eval_metrics] All PPL and FL = 0.0, GPT-2 Vietnamese model may not be working correctly"
        print(msg)
        eval_logger.warning(msg)
    else:
        msg = f"[eval_metrics] PPL calculation completed: avg_ppl={avg_ppl:.2f}, avg_fl={avg_fl:.4f}"
        print(msg)
        eval_logger.info(msg)
    
    return {"ppl": avg_ppl, "fl": avg_fl}

# --- Metric: STA (Toxicity Drop) using PhoBERT ---
def load_phobert_classifier(model_path: Optional[str] = None) -> tuple[Optional[AutoTokenizer], Optional[AutoModelForSequenceClassification]]:
    """Load PhoBERT classifier model to calculate toxicity probability."""
    global _phobert_model, _phobert_tokenizer
    
    if _phobert_model is not None and _phobert_tokenizer is not None:
        return _phobert_tokenizer, _phobert_model
    
    if model_path is None:
        # Tìm model mới nhất trong thư mục phobert_classifier
        current_dir = os.path.dirname(os.path.abspath(__file__))
        phobert_dir = os.path.abspath(
            os.path.join(current_dir, "../phobert_classifier")
        )
        if not os.path.exists(phobert_dir):
            eval_logger = get_eval_logger()
            eval_logger.warning(f"[eval_metrics] phobert_classifier directory not found: {phobert_dir}")
            return None, None
        
        model_dirs = [
            d for d in os.listdir(phobert_dir)
            if os.path.isdir(os.path.join(phobert_dir, d)) and d.startswith("phobert_classifier_model_")
        ]
        if not model_dirs:
            eval_logger = get_eval_logger()
            eval_logger.warning("[eval_metrics] PhoBERT classifier model not found")
            return None, None
        # Lấy model mới nhất (sort theo tên)
        model_dirs.sort(reverse=True)
        model_path = os.path.join(phobert_dir, model_dirs[0])
    
    if not os.path.exists(model_path):
        eval_logger = get_eval_logger()
        eval_logger.warning(f"[eval_metrics] Model path does not exist: {model_path}")
        return None, None
    
    try:
        eval_logger = get_eval_logger()
        eval_logger.info(f"[eval_metrics] Loading PhoBERT classifier from {model_path}...")
        print(f"[eval_metrics] Loading PhoBERT classifier from {model_path}...")
        _phobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _phobert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _phobert_model.to(_phobert_device)
        _phobert_model.eval()
        eval_logger.info("[eval_metrics] PhoBERT classifier loaded successfully")
        print("[eval_metrics] PhoBERT classifier loaded successfully")
        return _phobert_tokenizer, _phobert_model
    except Exception as exc:
        eval_logger = get_eval_logger()
        eval_logger.warning(f"[eval_metrics] Could not load PhoBERT classifier: {exc}")
        print(f"[eval_metrics] Could not load PhoBERT classifier: {exc}")
        return None, None


def get_toxicity_probability(text: str, tokenizer: Optional[AutoTokenizer] = None, model: Optional[AutoModelForSequenceClassification] = None) -> float:
    """
    Calculate toxicity probability (0-1) of text using PhoBERT classifier.
    Returns probability of "unsafe" class (1 = unsafe, 0 = safe).
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_phobert_classifier()
        if tokenizer is None or model is None:
            return 0.0
    
    if not text or not text.strip():
        return 0.0
    
    try:
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(_phobert_device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            
            # Lấy probability của class "unsafe" (class 1)
            # Giả sử model có 2 classes: 0 = safe, 1 = unsafe
            probs = torch.softmax(logits, dim=-1)
            # Lấy probability của class 1 (unsafe)
            unsafe_prob = probs[0][1].item()
            return float(unsafe_prob)
    except Exception as exc:
        eval_logger = get_eval_logger()
        eval_logger.warning(f"[eval_metrics] Error calculating toxicity for text '{text[:50]}...': {exc}")
        return 0.0


def calculate_sta_with_phobert(
    predictions: List[str],
    originals: List[str],
    phobert_model_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate STA based on PhoBERT classifier model.
    Uses toxicity probability (float 0-1) instead of binary.
    Returns ΔTox (Toxicity Drop) = avg(original_score - prediction_score).
    Only calculates tox_drop on sentences that are actually toxic (original_score > 0.5).
    """
    eval_logger = get_eval_logger()
    if not predictions:
        return {
            "sta": 0.0,
            "avg_tox_drop": 0.0,
            "avg_tox_drop_on_toxic_only": 0.0,
            "avg_toxicity_reduction": 0.0,
            "avg_original_toxicity": 0.0,
            "avg_prediction_toxicity": 0.0,
            "toxic_originals": 0,
            "toxic_predictions": 0,
        }
    
    # Load PhoBERT model
    tokenizer, model = load_phobert_classifier(phobert_model_path)
    if tokenizer is None or model is None:
        eval_logger.warning("[eval_metrics] PhoBERT classifier not available, returning default metrics")
        print("[eval_metrics] PhoBERT classifier not available, returning default metrics")
        return {
            "sta": 0.0,
            "avg_tox_drop": 0.0,
            "avg_tox_drop_on_toxic_only": 0.0,
            "avg_toxicity_reduction": 0.0,
            "avg_original_toxicity": 0.0,
            "avg_prediction_toxicity": 0.0,
            "toxic_originals": 0,
            "toxic_predictions": 0,
        }
    
    original_scores = []
    prediction_scores = []
    
    eval_logger.info("[eval_metrics] Calculating toxicity scores using PhoBERT classifier...")
    print("[eval_metrics] Calculating toxicity scores using PhoBERT classifier...")
    for orig, pred in tqdm(zip(originals, predictions), total=len(predictions), desc="Toxicity scoring"):
        orig_score = get_toxicity_probability(orig, tokenizer, model)
        pred_score = get_toxicity_probability(pred, tokenizer, model)
        
        original_scores.append(orig_score)
        prediction_scores.append(pred_score)
    
    # Tính ΔTox (Toxicity Drop) trên tất cả câu
    reductions = [o - p for o, p in zip(original_scores, prediction_scores)]
    avg_tox_drop = float(np.mean(reductions))
    
    # Tính ΔTox chỉ trên những câu thực sự độc (original_score > 0.5)
    toxic_reductions = [
        o - p for o, p in zip(original_scores, prediction_scores) if o > 0.5
    ]
    avg_tox_drop_on_toxic_only = (
        float(np.mean(toxic_reductions)) if toxic_reductions else 0.0
    )
    
    avg_original = float(np.mean(original_scores))
    avg_prediction = float(np.mean(prediction_scores))
    
    # Đếm số câu toxic (threshold = 0.5)
    toxic_originals = sum(1 for score in original_scores if score > 0.5)
    toxic_predictions = sum(1 for score in prediction_scores if score > 0.5)
    
    eval_logger.info(f"[eval_metrics] STA calculation completed: avg_tox_drop={avg_tox_drop:.4f}, toxic_originals={toxic_originals}")
    print(f"[eval_metrics] STA calculation completed: avg_tox_drop={avg_tox_drop:.4f}, toxic_originals={toxic_originals}")
    
    return {
        "sta": avg_tox_drop,  # STA is now ΔTox (all sentences)
        "avg_tox_drop": avg_tox_drop,  # New clearer name
        "avg_tox_drop_on_toxic_only": avg_tox_drop_on_toxic_only,  # Only on toxic sentences
        "avg_toxicity_reduction": avg_tox_drop,  # Alias
        "avg_original_toxicity": avg_original,
        "avg_prediction_toxicity": avg_prediction,
        "toxic_originals": toxic_originals,
        "toxic_predictions": toxic_predictions,
    }


def calculate_j(sim: float, fl: float, tox_drop: float) -> float:
    """
    Calculate Joint score (J) = sim * fl * normalized_tox_drop.
    
    Args:
        sim: Semantic similarity score (0-1)
        fl: Fluency score (0-1)
        tox_drop: Toxicity drop (ΔTox), can be positive or negative
    
    Returns:
        Joint score. If tox_drop < 0 (rewritten sentence is more toxic than original) then J = 0.
    """
    # If tox_drop < 0, rewritten sentence is more toxic than original -> J = 0
    if tox_drop < 0:
        return 0.0
    
    # Normalize tox_drop to [0, 1]
    # Assume max tox_drop is 1.0 (from 1.0 -> 0.0)
    normalized_tox_drop = min(max(tox_drop, 0.0), 1.0)
    
    return float(sim * fl * normalized_tox_drop)

# --- Main Evaluation Wrapper ---
def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    originals: List[str],
    original_items: Optional[List[Dict[str, Any]]], # For backward compatibility with old interface
    example_records: Sequence[Dict[str, Any]],
    phobert_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Central evaluation function.
    Calculates all metrics: BLEU, SIM, FL/PPL, STA (Toxicity Drop), and J-score.
    """
    eval_logger = get_eval_logger()
    eval_logger.info("[eval_metrics] Starting evaluate_predictions...")
    print("[eval_metrics] Starting evaluate_predictions...")
    
    # 1. Calculate BLEU (Reference-based)
    eval_logger.info("[eval_metrics] Calculating BLEU...")
    bleu = calculate_bleu_scores(predictions, references)
    
    # 2. Calculate SIM (Semantic Similarity)
    eval_logger.info("[eval_metrics] Calculating SIM...")
    sim_scores = calculate_sim(predictions, originals)
    
    # 3. Calculate Fluency & PPL (Language Model based)
    eval_logger.info("[eval_metrics] Calculating PPL and Fluency...")
    print("[eval_metrics] Calculating PPL and Fluency...")
    fl_metrics = calculate_perplexity_and_fluency(predictions)
    eval_logger.info(f"[eval_metrics] PPL/FL results: {fl_metrics}")
    print(f"[eval_metrics] PPL/FL results: {fl_metrics}")
    
    # 4. Calculate STA (Toxicity Drop) with PhoBERT
    eval_logger.info("[eval_metrics] Calculating STA (Toxicity Drop)...")
    print("[eval_metrics] Calculating STA (Toxicity Drop)...")
    sta_metrics = calculate_sta_with_phobert(
        predictions=predictions,
        originals=originals,
        phobert_model_path=phobert_model_path,
    )
    
    # 5. Calculate J-score (Joint score)
    sim_value = sim_scores.get("sim", 0.0)
    fl_value = fl_metrics.get("fl", 0.0)
    tox_drop = sta_metrics.get("avg_tox_drop", 0.0)
    j_score = calculate_j(sim_value, fl_value, tox_drop)
    
    eval_logger.info(f"[eval_metrics] J-score: {j_score:.6f}")
    print(f"[eval_metrics] J-score: {j_score:.6f}")
    
    # 6. Prepare examples for logging
    examples = []
    for idx, (record, pred) in enumerate(zip(example_records, predictions)):
        example = {
            "index": idx,
            "id": record.get("id", idx),
            "original": record.get("original", ""),
            "reference": record.get("reference", ""),
            "prediction": pred,
        }
        examples.append(example)

    # 7. Aggregate results with structure compatible with baseline scripts
    result = {
        "j": j_score,
        "sta": sta_metrics["sta"],
        "avg_tox_drop": sta_metrics["avg_tox_drop"],
        "avg_tox_drop_on_toxic_only": sta_metrics["avg_tox_drop_on_toxic_only"],
        "avg_original_toxicity": sta_metrics["avg_original_toxicity"],
        "avg_prediction_toxicity": sta_metrics["avg_prediction_toxicity"],
        "toxic_originals": sta_metrics["toxic_originals"],
        "toxic_predictions": sta_metrics["toxic_predictions"],
        "reference_based_metrics": {"bleu": bleu},
        "reference_free_metrics": {
            "sim": sim_scores,      # Dict {'sim': float}
            "fl": fl_metrics,       # Dict {'ppl': float, 'fl': float}
        },
        "total_examples": len(predictions),
        "examples": examples,
    }
    
    return result