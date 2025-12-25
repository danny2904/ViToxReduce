#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for ViToxReduce Pipeline
"""

from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent
with open(root / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(root / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vitoxreduce",
    version="1.0.2",
    author="joshswift294",
    author_email="joshswift294@gmail.com",
    description="A comprehensive pipeline for reducing toxicity in text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danny2904/vitoxreduce",
    license="Apache-2.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vitoxreduce=scripts.run_pipeline:main",
        ],
    },
)

