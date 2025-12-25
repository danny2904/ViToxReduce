# ViToxRewrite Dataset

This directory contains the ViToxRewrite dataset for training and evaluation.

## Dataset Files

- `vitoxrewrite_train.jsonl` - Training set
- `vitoxrewrite_validation.jsonl` - Validation set
- `vitoxrewrite_test.jsonl` - Test set

## Dataset Format

Each line in the JSONL files is a JSON object with the following structure:

```json
{
    "id": 0,
    "source": "ViHOS",
    "comment": "Original sentence text",
    "label": "safe" | "unsafe",
    "spans": ["list", "of", "toxic", "spans"],
    "unsafe_spans_indices": [[start1, end1], [start2, end2], ...],
    "rewrites": "Rewritten sentence (reference)",
    "reason": "Explanation of the rewrite"
}
```

### Fields

- `id`: Unique identifier for the example
- `source`: Source dataset name
- `comment`: Original sentence text (input to the pipeline)
- `label`: Safety label ("safe" or "unsafe")
- `spans`: List of toxic span texts
- `unsafe_spans_indices`: List of [start, end] character indices for toxic spans
- `rewrites`: Rewritten sentence (used as reference for evaluation)
- `reason`: Explanation of the rewrite

## Usage

### Using with the Pipeline

To process the test set:

```bash
python scripts/run_pipeline.py \
    --input dataset/vitoxrewrite_test.jsonl \
    --rewriter_model <path_to_bartpho_model> \
    --span_locator_model <path_to_span_locator_model> \
    --toxicity_detector_model <path_to_toxicity_classifier_model> \
    --output results.json
```

### Using in Python

```python
import json

# Load dataset
with open('dataset/vitoxrewrite_test.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        comment = item['comment']
        rewrites = item['rewrites']
        # Process with pipeline...
```

## Statistics

- **Training set**: ~8,844 examples
- **Validation set**: ~1,100 examples
- **Test set**: ~1,101 examples

## Notes

- The `comment` field is used as input to the pipeline
- The `rewrites` field is used as reference for evaluation metrics
- Safe examples have empty `spans` and `unsafe_spans_indices` arrays
- The `rewrites` field may be empty for safe examples

