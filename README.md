# fujielab-asr-parallel-cbs

Parallel CBS Transformer based ASR system built on ESPnet

## Overview
fujielab-asr-parallel-cbs is an automatic speech recognition (ASR) system based on ESPnet, featuring a parallelizable Contextual Block Streaming (CBS) Transformer.

## Features
- Implementation of a parallel CBS Transformer encoder extending the ESPnet framework
- Supports online and streaming ASR inference
- Pretrained models available via [Hugging Face](https://huggingface.co/)
- Example script for chunk-by-chunk streaming recognition

## Requirements
- Python 3.11
- `torch`, `torchaudio`, and other packages listed in `requirements.txt`

## Installation

### PyPI Installation
You can install the package directly from PyPI:
```bash
pip install fujielab-asr-parallel-cbs
```

### Local Installation
1. Install the dependencies and this package:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
   If you skip installing the package, running the examples may fail with
   `ModuleNotFoundError`. Alternatively, you can set `PYTHONPATH=$(pwd)` before
   executing the scripts.
2. If there are additional dependencies, please refer to `pyproject.toml`.

## Usage
### Example: Running Inference
You can perform inference from an audio file using `examples/run_streaming_asr.py`:

```bash
python examples/run_streaming_asr.py
```

The script streams the input audio in 100&nbsp;ms chunks and prints partial
results. At the first run it downloads a small example audio (`aps-smp.mp3`)
from the CSJ corpus and a pretrained model from Hugging Face. When the
recognition finishes successfully, the final transcript is displayed at the end
of the log.


## Directory Structure
- `fujielab/asr_parallel_cbs/espnet_ext/` : ESPnet extension implementation
  - `espnet/` : Extensions for ESPnet1
  - `espnet2/` : Extensions for ESPnet2 (ASR, transducer, joint network, etc.)
- `examples/` : Sample audio and inference scripts
- `warprnnt_pytorch/` : Dummy module for warprnnt_pytorch

## License
This repository is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements
This project is based on the ESPnet framework and incorporates contributions from various open-source projects. We thank the ESPnet team and contributors for their work.

