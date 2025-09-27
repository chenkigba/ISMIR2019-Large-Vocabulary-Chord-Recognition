# Installation Guide

## Installing from Git Repository

You can install this package directly from the git repository using pip:

```bash
# Install from GitHub (replace with your actual repository URL)
pip install git+https://github.com/your-username/ISMIR2019-Large-Vocabulary-Chord-Recognition.git

# Or install from a specific branch/tag
pip install git+https://github.com/your-username/ISMIR2019-Large-Vocabulary-Chord-Recognition.git@main

# Install in development mode (editable install)
pip install -e git+https://github.com/your-username/ISMIR2019-Large-Vocabulary-Chord-Recognition.git#egg=LVCR_ismir2019
```

## Usage

### Command Line Interface

After installation, you can use the chord recognition tool from the command line:

```bash
chord-recognition input_audio.mp3 output_chords.lab
chord-recognition input_audio.mp3 output_chords.lab submission
```

### Python API

You can also use the package in your Python code:

```python
import LVCR_ismir2019

# Using the main function
LVCR_ismir2019.chord_recognition_main(
    "input_audio.mp3", 
    "output_chords.lab", 
    "submission"
)

# Or import specific modules
from LVCR_ismir2019 import mir, extractors, io_new
```

## Requirements

- Python 3.7 or higher
- PyTorch 1.4.0 or higher
- See requirements.txt for full list of dependencies

## Pretrained Models

The package includes pretrained models in the `cache_data/` directory. These models will be automatically used for chord recognition.
