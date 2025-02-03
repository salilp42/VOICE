# VOICE 🎤 

Voice-based Parkinson's Disease Detection Framework

## Overview

VOICE is a comprehensive framework for processing and analyzing voice recordings to detect early signs of Parkinson's Disease (PD). The framework implements a complete pipeline from raw audio preprocessing to feature extraction and model training.

## Features 🔍

- **Robust Audio Preprocessing**
  - Automated voice activity detection
  - Silence removal
  - Audio segmentation with configurable overlap
  - Power normalization and DC offset removal

- **Multi-Dataset Support**
  - KCL dataset processing
  - Italian dataset processing
  - Extensible to other datasets

- **Standardized Output Format**
  - Numpy arrays for processed segments
  - Comprehensive metadata tracking
  - Consistent file structure

## Directory Structure 📁

```
VOICE/
├── src/
│   ├── preprocess_data.py      # Main preprocessing script
│   ├── data/
│   │   └── dataset.py          # Dataset handling utilities
│   └── models/                 # Model implementations
├── requirements.txt            # Project dependencies
└── README.md                  # This file
```

## Installation 💻

1. Clone the repository:
```bash
git clone https://github.com/salilp42/VOICE.git
cd VOICE
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation 🔧

Organize your audio files in the following structure:
```
data_directory/
├── participant_id_1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── participant_id_2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

## Usage 🚀

### 1. Preprocessing Audio Data

```bash
python src/preprocess_data.py --data-dir /path/to/data/directory --dataset [KCL|Italian]
```

This will:
- Process all WAV files in the specified directory
- Apply voice activity detection
- Generate 2-second segments with 50% overlap
- Save processed data in `Processed_Data_Complete/`

### 2. Output Structure

The preprocessing creates:
```
Processed_Data_Complete/
├── raw_segments/
│   └── {dataset}_segments.npy      # Processed audio segments
└── {dataset}_metadata.csv          # Segment metadata
```

### 3. Configuration

Default preprocessing parameters:
- Sampling rate: 16kHz
- Segment length: 2 seconds (32000 samples)
- Segment overlap: 50%
- Minimum segments per file: 3
- Voice activity detection thresholds:
  - Energy threshold: 0.1
  - Zero-crossing rate threshold: 0.2

To modify these parameters, edit the corresponding values in `src/preprocess_data.py`.

## Requirements 📋

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Librosa >= 0.9.0
- SoundFile >= 0.10.3
- tqdm >= 4.62.0

## Author

Salil Patel

## License

MIT License 