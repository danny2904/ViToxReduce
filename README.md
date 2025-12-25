# ViToxReduce Pipeline

A comprehensive pipeline for reducing toxicity in text using a streamlined 3-tier architecture.

![Pipeline Architecture](images/architecture_diagram.png)

## 📋 Description

ViToxReduce is an automated system for reducing toxicity in text using a 3-tier architecture:
1. **Safety Checker**: Detects toxic sentences
2. **Span Tagger**: Locates toxic phrases
3. **Contextual Rewriter**: Safely rewrites sentences

## 🏗️ System Architecture

The ViToxReduce pipeline uses a streamlined 3-tier architecture to efficiently process and detoxify text:

![Pipeline Architecture](images/architecture_diagram.png)

### Example Flow

The following diagram shows an example of how the pipeline processes a toxic sentence:

![Example Flow](images/example_flow.png)

### Tier 1: Safety Checker
- **Function**: Classifies input sentences as `safe` (safe) or `unsafe` (toxic)
- **Model**: PhoBERT (Sequence Classification)
- **Processing Flow**:
  - If `Safe` → Returns the original sentence unchanged, terminates processing (saves computational resources for ~50% of non-toxic sentences)
  - If `Unsafe` → Proceeds to Tier 2 for further processing

### Tier 2: Span Tagger
- **Function**: Scans sentences and labels each word to identify toxic phrase locations (Toxic Spans)
- **Model**: PhoBERT (Token Classification / BIO Tagging)
- **Output**: A list of spans `S = [s_1, s_2, ...]` (character indices)
- **Role**: Provides "hints" to Tier 3 about which parts to focus on, preventing over-editing of safe regions

### Tier 3: Contextual Rewriter
- **Function**: Generates new sentences that remove or replace toxic spans while preserving context and intent of remaining parts
- **Model**: BARTpho (Seq2Seq - Span-guided)
- **Input**: Original sentence `X` + Span list `S` (combined into prompt: "Original sentence: X. Words to fix: S")
- **Generation Mechanism**: Uses One-time Rewriting with Beam Search to select the best result (Top-1) without iterative checking, improving processing speed

## 📦 Installation

### System Requirements
- Python >= 3.7
- CUDA (optional, for GPU acceleration)

### Install via pip

```bash
pip install vitoxreduce
```

*(Dev mode)* From source:
```bash
pip install -r requirements.txt
pip install -e .
```

### Required Models
You need 3 fine-tuned models (can be loaded directly from Hugging Face Hub):
1. **BARTpho Rewriter** (Tier 3) — `joshswift/bartpho-rewriter`
2. **PhoBERT Span Locator** (Tier 2) — `joshswift/phobert-span`
3. **PhoBERT Toxicity Classifier** (Tier 1) — `joshswift/phobert-toxicity`

You can use the repo IDs directly (online), or download locally:
```bash
pip install huggingface_hub
huggingface-cli login          # if private or rate-limited
huggingface-cli download joshswift/bartpho-rewriter   --local-dir ./models/rewriter
huggingface-cli download joshswift/phobert-span       --local-dir ./models/span
huggingface-cli download joshswift/phobert-toxicity   --local-dir ./models/toxicity
```

### Dataset (optional, for testing/eval)
- Hugging Face dataset: `joshswift/vitoxrewrite`
- Download locally (JSONL kept intact):
```bash
pip install huggingface_hub
huggingface-cli login  # if needed
huggingface-cli download joshswift/vitoxrewrite --local-dir ./dataset
```
After download, you will have:
- `dataset/vitoxrewrite_train.jsonl`
- `dataset/vitoxrewrite_validation.jsonl`
- `dataset/vitoxrewrite_test.jsonl`

## 🚀 Quick Start

### Option 1: Automated Setup Script (Recommended for First-Time Users)

The easiest way to get started is using the automated setup script that installs the package, downloads models, and runs a test:

```bash
# Clone or download the repository
cd vitoxreduce_pipeline_github

# Run the automated setup script
python examples/example_usage.py
```

**What the script does:**
- Installs `vitoxreduce` package from PyPI
- Downloads 3 required models from Hugging Face (skips if already exists)
- Runs a smoke test with sample text
- Saves results to `./results/smoke_test_TIMESTAMP.json`

**Customize the script:**
```bash
# Use custom model directory
python examples/example_usage.py --models-dir ./my_models

# Use Hugging Face token (if repos are private/rate-limited)
python examples/example_usage.py --token hf_xxxxxxxxxxxxx

# Custom output file
python examples/example_usage.py --output ./my_results.json
```

### Option 2: Manual Setup (Step-by-Step)

If you prefer to set up manually or need more control, follow these steps:

#### 1) Install Package
```bash
pip install vitoxreduce
```

#### 2) Download Models (Optional - can use online models instead)
```bash
pip install huggingface_hub
huggingface-cli download joshswift/bartpho-rewriter   --local-dir ./models/rewriter
huggingface-cli download joshswift/phobert-span       --local-dir ./models/span
huggingface-cli download joshswift/phobert-toxicity   --local-dir ./models/toxicity
```

#### 3) Run CLI
```bash
# Single sentence (using online models)
vitoxreduce \
  --input "Từ lúc mấy bro cmt cực kì cl gì đấy..." \
  --rewriter_model joshswift/bartpho-rewriter \
  --span_locator_model joshswift/phobert-span \
  --toxicity_detector_model joshswift/phobert-toxicity \
  --output result.json \
  --verbose

# Or use local models
vitoxreduce \
  --input "Your text here" \
  --rewriter_model ./models/rewriter \
  --span_locator_model ./models/span \
  --toxicity_detector_model ./models/toxicity \
  --output result.json \
  --verbose
```

#### 4) Python API
```python
from vitoxreduce import ViToxReducePipeline

pipeline = ViToxReducePipeline(
    rewriter_model_path="joshswift/bartpho-rewriter",  # or local path
    span_locator_model_path="joshswift/phobert-span",
    toxicity_detector_model_path="joshswift/phobert-toxicity",
    num_beams=5,
)

result = pipeline.process("Your text here", verbose=True)
print("Rewritten:", result["rewritten"])
print("Safe?:", result["rewritten_is_safe"])
print("Toxicity:", result["toxicity_score"], "→", result["rewritten_toxicity_score"])
```

## ⚙️ Command-Line Arguments

### Required Arguments
- `--rewriter_model`: Path to BARTpho rewriter model (Tier 3) or Hugging Face repo ID
- `--span_locator_model`: Path to PhoBERT span locator model (Tier 2) or Hugging Face repo ID
- `--toxicity_detector_model`: Path to PhoBERT toxicity classifier (Tier 1) or Hugging Face repo ID
- `--input`: Input text, text file (one sentence per line), or JSONL file with 'comment' field

### Optional Arguments
- `--output`: Output JSON file path (default: auto-generated or stdout)
- `--num_beams`: Number of beams for beam search (default: 5)
- `--toxicity_threshold`: Threshold for unsafe classification (default: 0.5)
- `--device`: Device to use - `cuda`, `cpu`, or `auto` (default: auto)
- `--verbose`: Print detailed processing information
- `--mode`: Processing mode - `auto`, `single`, `file`, or `jsonl` (default: auto)

## 📊 Output Format

The pipeline returns a dictionary with the following keys:

```python
{
    "original": "Original sentence",
    "is_safe": False,                    # True if safe, False if unsafe
    "toxicity_score": 0.85,              # Toxicity score (0.0-1.0)
    "spans": [(10, 15), (20, 25)],       # List of detected spans (character indices)
    "span_texts": ["toxic", "word"],     # List of span texts
    "rewritten": "Rewritten sentence",   # Rewritten sentence (or original if safe)
    "rewritten_is_safe": True,           # True if rewritten sentence is safe
    "rewritten_toxicity_score": 0.15,    # Toxicity score of rewritten sentence
    "processing_time": 1.2                # Processing time (seconds)
}
```

## 📁 Project Structure

```
vitoxreduce_pipeline_github/
├── vitoxreduce/                    # Main package
│   ├── pipeline.py                 # Main 3-tier pipeline
│   ├── tier1_toxicity_detector.py # Tier 1: Safety Checker
│   ├── tier2_span_locator.py      # Tier 2: Span Tagger
│   ├── tier3_rewrite_generator.py # Tier 3: Contextual Rewriter
│   └── ...
├── scripts/                        # CLI scripts
│   └── run_pipeline.py            # Main CLI entry point
├── examples/                       # Usage examples
│   └── example_usage.py           # Automated setup & test script
├── dataset/                        # Sample dataset (optional)
└── requirements.txt                # Python dependencies
```

## 🔍 Evaluation Metrics

The pipeline supports the following metrics (when reference is available):

- **BLEU**: Measures similarity with reference
- **SIM**: Semantic similarity (using SBERT)
- **FL**: Fluency score (based on GPT-2 PPL)
- **STA**: Toxicity Drop (toxicity reduction)
- **J-score**: Joint score = SIM × FL × normalized_tox_drop

### Performance Comparison

![Performance Comparison](images/performance_comparison.png)

The above chart shows ViToxReduce's performance compared to baseline methods across key metrics including BLEU, SIM, J-score, and toxicity reduction.

## 📖 API Documentation

### ViToxReducePipeline

Main pipeline class for processing text.

#### `__init__(self, toxicity_detector_model_path, span_locator_model_path, rewriter_model_path, ...)`

Initialize the pipeline with model paths.

**Parameters:**
- `toxicity_detector_model_path` (str, required): Path to PhoBERT toxicity classifier
- `span_locator_model_path` (str, required): Path to PhoBERT span locator model
- `rewriter_model_path` (str, required): Path to BARTpho rewriter model
- `toxicity_threshold` (float, optional): Threshold for unsafe classification (default: 0.5)
- `num_beams` (int, optional): Number of beams for beam search (default: 5)
- `device` (str, optional): Device to use (cuda/cpu, default: auto)

#### `process(self, text, verbose=False)`

Process a single sentence through the 3-tier pipeline.

**Parameters:**
- `text` (str): Sentence to process
- `verbose` (bool): Print detailed processing information

**Returns:**
- `dict`: Processing result with keys: `original`, `is_safe`, `toxicity_score`, `spans`, `span_texts`, `rewritten`, `rewritten_is_safe`, `rewritten_toxicity_score`, `processing_time`

#### `process_batch(self, texts, verbose=False)`

Process a batch of sentences.

**Parameters:**
- `texts` (List[str]): List of sentences to process
- `verbose` (bool): Print detailed processing information

**Returns:**
- `List[dict]`: List of processing results

## ⚠️ Important Notes

1. **Model Paths**: All 3 model paths are required. You can use Hugging Face repo IDs (e.g., `joshswift/bartpho-rewriter`) or local paths.

2. **Tier 1 Optimization**: If a sentence is detected as `safe`, the pipeline returns the original sentence and skips Tier 2 & 3, saving computational resources.

3. **GPU/CPU Auto-Detection**: The pipeline automatically detects and uses GPU if available. If no GPU is found, it automatically falls back to CPU. Use `--device cpu` to force CPU mode or `--device cuda` to force GPU mode.

4. **Output Files**: Results are saved to JSON files containing original text, rewritten text, toxicity scores, spans, and processing statistics.

## 🐛 Troubleshooting

### Error: "Model not found"
- Check that model paths are correct
- Ensure models are trained and saved in the correct format

### Error: "Out of memory"
- Reduce `--num_beams`
- Use `--device cpu` if GPU runs out of memory
- Reduce batch size when processing

### Error: "Span Locator model not loaded"
- Ensure the correct path is specified with `--span_locator_model`

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version >= 3.7

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

Apache License 2.0

See [LICENSE](LICENSE) file for details.

## 👥 Authors

ViToxReduce Pipeline - Text Toxicity Reduction System (3-Tier Streamlined Architecture)
