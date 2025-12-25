#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run ViToxReduce pipeline
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from vitoxreduce import ViToxReducePipeline

# Import evaluate_predictions from eval_metrics
try:
    from vitoxreduce.eval_metrics import evaluate_predictions
except ImportError as e:
    # Logger will be setup later in main(), but need to log warning now
    import warnings
    warnings.warn(f"Could not import evaluate_predictions from eval_metrics: {e}. Metrics will not be calculated.")
    evaluate_predictions = None

# Import normalize_rewrite_field from bartpho_span_baseline to read rewrites field
try:
    from vitoxreduce.bartpho_span_baseline import normalize_rewrite_field
except ImportError:
    # Fallback function if import fails
    def normalize_rewrite_field(rewrites):
        if not rewrites:
            return ""
        if isinstance(rewrites, str):
            return rewrites.strip()
        if isinstance(rewrites, (list, tuple)):
            for candidate in rewrites:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return ""


def setup_logging(log_dir: str = "./logs"):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    log_file = os.path.join(log_dir, f'vitoxreduce_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


def load_jsonl(path: str) -> List[dict]:
    """Load data from JSONL file"""
    data = []
    if not os.path.exists(path):
        logger.warning(f"File does not exist: {path}")
        return data
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Run ViToxReduce pipeline")
    
    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file (JSONL with 'comment' field or text file, or direct text). If not specified, will automatically find test set."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: stdout or input_results.json)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "file", "jsonl"],
        default="auto",
        help="Processing mode (auto: auto-detect)"
    )
    
    # Model paths
    parser.add_argument(
        "--toxicity_detector_model",
        type=str,
        default=None,
        help="Path to PhoBERT toxicity classifier (Tier 1 - Safety Checker)"
    )
    parser.add_argument(
        "--span_locator_model",
        type=str,
        default=None,
        help="Path to PhoBERT Token Classification model (Tier 2 - Span Tagger). If not specified, will auto-detect."
    )
    parser.add_argument(
        "--rewriter_model",
        type=str,
        default=None,
        help="Path to BARTpho rewriter model (Tier 3 - Contextual Rewriter). If not specified, will auto-detect."
    )
    
    # Config
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search in Tier 3 (default: 5)"
    )
    parser.add_argument(
        "--toxicity_threshold",
        type=float,
        default=0.5,
        help="Threshold for unsafe classification (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging()
    
    # Check required model paths
    if args.rewriter_model is None:
        logger.error("--rewriter_model is required! Please specify the path to BARTpho rewriter model.")
        sys.exit(1)
    
    if args.span_locator_model is None:
        logger.error("--span_locator_model is required! Please specify the path to PhoBERT span locator model.")
        sys.exit(1)
    
    if args.toxicity_detector_model is None:
        logger.error("--toxicity_detector_model is required! Please specify the path to PhoBERT toxicity classifier model.")
        sys.exit(1)
    
    if args.input is None:
        logger.error("--input is required! Please specify input file or text to process.")
        sys.exit(1)
    
    # Initialize pipeline
    logger.info("Initializing ViToxReduce pipeline (3 tiers)...")
    try:
        pipeline = ViToxReducePipeline(
            toxicity_detector_model_path=args.toxicity_detector_model,
            toxicity_threshold=args.toxicity_threshold,
            span_locator_model_path=args.span_locator_model,
            rewriter_model_path=args.rewriter_model,
            num_beams=args.num_beams,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Xác định mode
    if args.mode == "auto":
        if os.path.isfile(args.input):
            if args.input.endswith('.jsonl'):
                mode = "jsonl"
            else:
                mode = "file"
        else:
            mode = "single"
    else:
        mode = args.mode
    
    # Process input
    texts = []
    input_data = []  # Store all data to get reference later
    if mode == "single":
        # Direct text
        texts = [args.input]
        input_data = [{"comment": args.input}]
    elif mode == "file":
        # Text file, each line is a sentence
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
            input_data = [{"comment": text} for text in texts]
    elif mode == "jsonl":
        # JSONL file
        input_data = load_jsonl(args.input)
        # Filter and map only items with comment
        texts = []
        filtered_input_data = []
        for item in input_data:
            comment = item.get('comment', '').strip()
            if comment:
                texts.append(comment)
                filtered_input_data.append(item)
        input_data = filtered_input_data  # Keep only items with comment
    
    if not texts:
        logger.error("No text to process!")
        sys.exit(1)
    
    logger.info(f"Processing {len(texts)} sentences...")
    
    # Process
    pipeline_results = pipeline.process_batch(texts, verbose=args.verbose)
    
    # Prepare result structure in sample format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for metrics calculation
    predictions = []
    references = []
    originals = []
    example_records = []  # To pass to evaluate_predictions
    
    # Convert pipeline results to sample format and prepare data for metrics
    examples = []
    for idx, result in enumerate(pipeline_results):
        original = result.get('original', '')
        prediction = result.get('rewritten', result.get('original', ''))
        
        # Get reference from input_data if available
        reference = None
        example_id = idx
        if mode == "jsonl" and idx < len(input_data):
            input_item = input_data[idx]
            if 'id' in input_item:
                example_id = input_item['id']
            # Prioritize reading from 'rewrites' field (same as baseline)
            if 'rewrites' in input_item:
                reference = normalize_rewrite_field(input_item['rewrites'])
            elif 'reference' in input_item:
                reference = input_item['reference']
            elif 'rewritten' in input_item:
                reference = input_item['rewritten']
        
        # If no reference, use original as reference (fallback)
        if reference is None or not reference.strip():
            reference = original
        
        example = {
            "index": idx,
            "original": original,
            "prediction": prediction,
            "is_safe": result.get('is_safe', False),
            "toxicity_score": result.get('toxicity_score', 0.0),
            # NOTE: spans and span_texts only have values when unsafe (tier 2 has run)
            # For safe: spans = [], span_texts = [] (because tier 2 is skipped)
            "spans": result.get('spans', []),  # Character indices from Tier 2 (only when unsafe)
            "span_texts": result.get('span_texts', []),  # Text of spans from Tier 2 (only when unsafe)
            # Re-check results after rewrite
            "rewritten_is_safe": result.get('rewritten_is_safe'),  # True if rewritten sentence is safe
            "rewritten_toxicity_score": result.get('rewritten_toxicity_score'),  # Toxicity score of rewritten sentence
            "processing_time": result.get('processing_time', 0.0)
        }
        
        # Add reference and id if available in input data (for JSONL)
        if mode == "jsonl" and idx < len(input_data):
            input_item = input_data[idx]
            if 'id' in input_item:
                example["id"] = input_item['id']
            # Add reference if available
            if 'reference' in input_item:
                example["reference"] = input_item['reference']
            elif 'rewritten' in input_item:
                example["reference"] = input_item['rewritten']
        
        examples.append(example)
        
        # Prepare data for metrics (only calculate for sentences with reference)
        predictions.append(prediction)
        references.append(reference)
        originals.append(original)
        example_records.append({
            "id": example_id,
            "original": original,
            "reference": reference,
            "prediction": prediction,
            "spans": result.get('spans', []),
            "span_texts": result.get('span_texts', [])
        })
    
    # Calculate metrics if evaluate_predictions is available
    metrics_result = None
    if evaluate_predictions is not None:
        logger.info("Calculating metrics (BLEU, SIM, FL, STA, J-score)...")
        try:
            # Get path to toxicity detector model to calculate STA
            phobert_model_path = args.toxicity_detector_model or None
            metrics_result = evaluate_predictions(
                predictions=predictions,
                references=references,
                originals=originals,
                original_items=None,  # Not needed
                example_records=example_records,
                phobert_model_path=phobert_model_path
            )
            logger.info("Metrics calculation completed!")
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            metrics_result = None
    
    # Create output structure in baseline format
    output_data = {
        "dataset": args.input if args.input else "stdin",
        "generated_at": timestamp,
        "pipeline": "vitoxreduce_3tier",
        "model_paths": {
            "toxicity_detector": args.toxicity_detector_model or "auto",
            "span_locator": args.span_locator_model or "auto",
            "rewriter": args.rewriter_model or "auto"
        },
        "config": {
            "toxicity_threshold": args.toxicity_threshold,
            "num_beams": args.num_beams,
            "device": args.device or "auto"
        },
        "total_toxic_examples": sum(1 for r in pipeline_results if not r.get('is_safe', True)),
    }
    
    # Add result object with metrics (same as baseline)
    if metrics_result is not None:
        # Update examples in metrics_result with spans from pipeline
        # metrics_result["examples"] was created by evaluate_predictions but doesn't have spans
        # Need to merge with examples from pipeline to get spans
        for idx, example in enumerate(examples):
            if idx < len(metrics_result["examples"]):
                # Add spans and other fields to example from metrics_result
                metrics_result["examples"][idx].update({
                    "spans": example.get("spans", []),
                    "span_texts": example.get("span_texts", []),
                    "is_safe": example.get("is_safe", False),
                    "toxicity_score": example.get("toxicity_score", 0.0),
                    "rewritten_is_safe": example.get("rewritten_is_safe"),
                    "rewritten_toxicity_score": example.get("rewritten_toxicity_score"),
                    "processing_time": example.get("processing_time", 0.0)
                })
        output_data["result"] = metrics_result
    else:
        # If metrics cannot be calculated, create empty result object
        output_data["result"] = {
            "j": 0.0,
            "sta": 0.0,
            "avg_tox_drop": 0.0,
            "avg_tox_drop_on_toxic_only": 0.0,
            "avg_original_toxicity": 0.0,
            "avg_prediction_toxicity": 0.0,
            "toxic_originals": 0,
            "toxic_predictions": 0,
            "reference_based_metrics": {"bleu": {"bleu": 0.0}},
            "reference_free_metrics": {
                "sim": {"sim": 0.0},
                "fl": {"ppl": 0.0, "fl": 0.0}
            },
            "total_examples": len(pipeline_results),
            "examples": examples
        }
    
    # Save results
    if args.output:
        output_path = args.output
    elif mode != "single":
        # Create result folder if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(args.input)) if os.path.isfile(args.input) else os.getcwd()
        result_dir = os.path.join(current_dir, "result")
        os.makedirs(result_dir, exist_ok=True)
        
        # Create output filename with timestamp
        base_name = os.path.basename(args.input)
        base_name_no_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(result_dir, f"{base_name_no_ext}_results_{timestamp}.json")
    else:
        output_path = None
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        # In ra stdout
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
    
    # Statistics
    safe_count = sum(1 for r in pipeline_results if r.get('is_safe', False))
    unsafe_count = len(pipeline_results) - safe_count
    rewritten_count = sum(1 for r in pipeline_results if not r.get('is_safe', True) and r.get('rewritten') != r.get('original'))
    avg_time = sum(r.get('processing_time', 0) for r in pipeline_results) / len(pipeline_results) if pipeline_results else 0
    
    # Statistics on rewrite effectiveness
    # Number of sentences successfully rewritten (from unsafe -> safe)
    successfully_detoxified = sum(1 for r in pipeline_results 
                                  if not r.get('is_safe', True) 
                                  and r.get('rewritten_is_safe') == True)
    # Number of sentences still unsafe after rewrite
    still_unsafe_after_rewrite = sum(1 for r in pipeline_results 
                                     if not r.get('is_safe', True) 
                                     and r.get('rewritten_is_safe') == False)
    # Calculate average toxicity reduction for rewritten sentences
    toxicity_reductions = []
    for r in pipeline_results:
        if not r.get('is_safe', True) and r.get('rewritten_toxicity_score') is not None:
            original_tox = r.get('toxicity_score', 0.0)
            rewritten_tox = r.get('rewritten_toxicity_score', 0.0)
            toxicity_reductions.append(original_tox - rewritten_tox)
    avg_toxicity_reduction = sum(toxicity_reductions) / len(toxicity_reductions) if toxicity_reductions else 0.0
    success_rate = (successfully_detoxified / unsafe_count * 100) if unsafe_count > 0 else 0.0
    
    logger.info(f"\n{'='*60}")
    logger.info("STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total sentences: {len(pipeline_results)}")
    logger.info(f"Safe (kept unchanged): {safe_count} ({safe_count/len(pipeline_results)*100:.1f}%)")
    logger.info(f"Unsafe (processed): {unsafe_count} ({unsafe_count/len(pipeline_results)*100:.1f}%)")
    logger.info(f"Rewritten: {rewritten_count} ({rewritten_count/len(pipeline_results)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"REWRITE EFFECTIVENESS:")
    logger.info(f"  - Successfully rewritten (unsafe -> safe): {successfully_detoxified} ({success_rate:.1f}%)")
    still_unsafe_rate = (still_unsafe_after_rewrite/unsafe_count*100) if unsafe_count > 0 else 0.0
    logger.info(f"  - Still unsafe after rewrite: {still_unsafe_after_rewrite} ({still_unsafe_rate:.1f}%)")
    logger.info(f"  - Average toxicity reduction: {avg_toxicity_reduction:.3f}")
    
    # Print metrics if available
    if metrics_result is not None:
        logger.info(f"")
        logger.info(f"METRICS:")
        logger.info(f"  - J-score: {metrics_result.get('j', 0.0):.6f}")
        logger.info(f"  - STA: {metrics_result.get('sta', 0.0):.6f}")
        logger.info(f"  - Avg Tox Drop: {metrics_result.get('avg_tox_drop', 0.0):.6f}")
        logger.info(f"  - BLEU: {metrics_result.get('reference_based_metrics', {}).get('bleu', {}).get('bleu', 0.0):.6f}")
        logger.info(f"  - SIM: {metrics_result.get('reference_free_metrics', {}).get('sim', {}).get('sim', 0.0):.6f}")
        logger.info(f"  - FL: {metrics_result.get('reference_free_metrics', {}).get('fl', {}).get('fl', 0.0):.6f}")
    
    logger.info(f"")
    logger.info(f"Average processing time: {avg_time:.3f}s per sentence")
    if output_path:
        logger.info(f"Results saved to: {output_path}")
    logger.info(f"Processing completed successfully!")


if __name__ == "__main__":
    main()
