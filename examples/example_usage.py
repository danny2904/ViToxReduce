#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end test script for vitoxreduce
- Installs vitoxreduce package from PyPI
- Downloads 3 models from Hugging Face
- Runs smoke test
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def print_info(msg):
    print(f"[INFO] {msg}")


def print_ok(msg):
    print(f"[OK] {msg}")


def print_warn(msg):
    print(f"[WARN] {msg}")


def print_err(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


def run_cmd(cmd, check=True):
    """Run shell command and return exit code"""
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print_err(f"Command failed: {cmd}")
        sys.exit(1)
    return result.returncode


def install_packages():
    """Install vitoxreduce and huggingface_hub"""
    print_info("Installing vitoxreduce and huggingface_hub...")
    run_cmd(f"{sys.executable} -m pip install --upgrade vitoxreduce huggingface_hub")
    print_ok("Packages installed")


def hf_login(token=None):
    """Login to Hugging Face if token provided"""
    if token:
        print_info("Logging in to Hugging Face...")
        run_cmd(f"huggingface-cli login --token {token} --add-to-git-credential")
    else:
        print_warn("No HF token provided. If repos are private/rate-limited, use --token <hf_xxx>")


def download_model(repo_id, target_dir):
    """Download model from Hugging Face, skip if already exists"""
    target_path = Path(target_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists (check for config.json and model file)
    config_file = target_path / "config.json"
    model_file = target_path / "model.safetensors"
    pytorch_model = target_path / "pytorch_model.bin"
    
    if config_file.exists() and (model_file.exists() or pytorch_model.exists()):
        print_ok(f"Model already exists at {target_dir}, skipping download")
        return
    
    print_info(f"Downloading {repo_id} -> {target_dir}")
    run_cmd(f"huggingface-cli download {repo_id} --local-dir {target_dir} --local-dir-use-symlinks False")
    print_ok(f"Downloaded to {target_dir}")


def run_smoke_test(rewriter_model, span_model, toxic_model, output_file=None):
    """Run smoke test with CLI"""
    print_info("Running smoke test...")
    test_input = "Từ lúc mấy bro cmt cực kì cl gì đấy..."
    
    # Create output file path if not provided
    if output_file is None:
        from datetime import datetime
        output_file = f"./results/smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f"vitoxreduce "
        f'--input "{test_input}" '
        f"--rewriter_model {rewriter_model} "
        f"--span_locator_model {span_model} "
        f"--toxicity_detector_model {toxic_model} "
        f"--output {output_file} "
        f"--verbose"
    )
    run_cmd(cmd)
    print_ok(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end test for vitoxreduce")
    parser.add_argument("--install-target", default="vitoxreduce", help="Package name to install (default: vitoxreduce)")
    parser.add_argument("--rewriter-repo", default="joshswift/bartpho-rewriter", help="Rewriter model repo ID")
    parser.add_argument("--span-repo", default="joshswift/phobert-span", help="Span locator model repo ID")
    parser.add_argument("--toxic-repo", default="joshswift/phobert-toxicity", help="Toxicity detector model repo ID")
    parser.add_argument("--token", default="", help="Hugging Face token (optional)")
    parser.add_argument("--models-dir", default="./models", help="Directory to save models (default: ./models)")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: ./results/smoke_test_TIMESTAMP.json)")
    args = parser.parse_args()

    # Check Python
    print_ok(f"Python: {sys.version}")

    # Install packages
    install_packages()

    # HF login
    if args.token:
        hf_login(args.token)
    else:
        hf_login()

    # Download models
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    rewriter_path = str(models_dir / "rewriter")
    span_path = str(models_dir / "span")
    toxic_path = str(models_dir / "toxicity")

    download_model(args.rewriter_repo, rewriter_path)
    download_model(args.span_repo, span_path)
    download_model(args.toxic_repo, toxic_path)

    # Run smoke test
    run_smoke_test(rewriter_path, span_path, toxic_path, args.output)

    print_ok(f"Done. Models at: {models_dir}. CLI test finished.")


if __name__ == "__main__":
    main()

