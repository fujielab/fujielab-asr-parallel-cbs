# fujielab-asr-parallel-cbs

Parallel CBS Transformer based ASR system built on ESPnet

## Overview
fujielab-asr-parallel-cbs is an automatic speech recognition (ASR) system based on ESPnet, featuring a parallelizable Contextual Block Streaming (CBS) Transformer.

## Features
- Implementation of a parallel CBS Transformer Encoder extending the ESPnet framework
- Supports online and streaming ASR inference

## Installation

### PyPI Installation
You can install the package directly from PyPI:
```bash
pip install fujielab-asr-parallel-cbs
```

### Local Installation
1. Install the required Python packages:
   ```bash
   pip install -e .
   ```
2. If there are additional dependencies, please refer to `pyproject.toml`.

## Usage
### Example: Running Inference
You can perform inference from an audio file using `examples/run_streaming_asr.py`:

```bash
python examples/run_streaming_asr.py
```

It will automatically download the pre-trained model from Hugging Face Hub
and sample audio files from CSJ (Corpus of Spontaneous Japanese) official site.


## Directory Structure
- `espnet_ext/` : ESPnet extension implementation
  - `espnet/` : Extensions for ESPnet1
  - `espnet2/` : Extensions for ESPnet2 (ASR, transducer, joint network, etc.)
- `examples/` : Sample audio and inference scripts
- `warprnnt_pytorch/` : Dummy module for warprnnt_pytorch

## License
This repository is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements
This project is based on the ESPnet framework and incorporates contributions from various open-source projects. We thank the ESPnet team and contributors for their work.

