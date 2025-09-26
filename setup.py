#!/usr/bin/env python3
"""
Setup script for ISMIR2019 Large Vocabulary Chord Recognition
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.MD", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chord-recognition-ismir2019",
    version="1.0.0",
    author="Junyan Jiang",
    author_email="",  # Add email if available
    description="Large-Vocabulary Chord Transcription via Chord Structure Decomposition",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ISMIR2019-Large-Vocabulary-Chord-Recognition",  # Update with actual repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Update based on your LICENSE file
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        "": ["data/*.txt", "data/*.csv", "data/*.lab", "data/*.n3", "data/*.svl", "data/*.pkl", "data/*.xlsx"],
        "cache_data": ["*.sdict"],
    },
    entry_points={
        "console_scripts": [
            "chord-recognition=chord_recognition:main",
        ],
    },
    zip_safe=False,
)
