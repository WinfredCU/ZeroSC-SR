# ZeroSC-SR_public

ZeroSC-SR: Zero-Shot Semantic Communication with Speech Reconstruction


## Demo

A demonstration of ZeroSC-SR is available here:  [**ZeroSC-SR Demo**](https://winfredcu.github.io/ZeroSC-SR_demo/#abstract)


## Repository Structure

```bash

ZeroSC-SR/
├── configs/                # Configuration files for models and experiments
├── data/                   # Placeholder for data or data loading scripts
├── models/                 # Model architectures and checkpoints
├── scripts/                # Auxiliary scripts (training, inference, evaluation, channel transmission)
├── utils/                  # Utility and helper functions
├── results/                # Generated results (logs, figures, etc.)
└── README.md               # This README
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WinfredCU/ZeroSC-SR.git
   cd ZeroSC-SR
   ```

2. **Install dependencies:**
   We recommend creating a new virtual environment before installing it.

   ```bash
   conda create -n ZeroSC-SR python=3.8
   conda activate ZeroSC-SR
   pip install -r requirements.txt
   ```

3. **Download or prepare data:**
   - Follow the instructions in the `data/` folder to set up datasets.


## Key Features

- **Frequency-Selective Fading Channels**: Employs a sort-match strategy to efficiently allocate channel resources.
- **High Compression Ratio**: Reduces transmission data size, beneficial in low-power and noisy conditions.
- **Digital Transmission**: Digital transmission of both phonemes and acoustic features over frequency-selective fading channels. 
- **Zero-Shot Capability**: Works without fine-tuning on new speakers. 

## Model Architecture and Parameters

| Component        | Layer Name                                                     | Type                       | Parameters                                                    |
| ---------------- | -------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------- |
| Semantic Encoder | ASR: Whisper – Encoder Transformer Module                      | Encoder Transformer Module | layers=24; d_model=1024; ffn=4096; heads=16                   |
| Semantic Encoder | ASR: Whisper – Decoder Transformer Module                      | Decoder Transformer Module | layers=24; d_model=1024; ffn=4096; heads=16                   |
| Semantic Encoder | Acoustic Encoder: EnCodec – Enc-front Module                   | 1-D Conv                   | channels=32; kernel=7                                         |
| Semantic Encoder | Acoustic Encoder: EnCodec – 4× Downsample Module               | Residual Conv              | kernel=3; strides=2 / 4 / 5 / 8                               |
| Semantic Encoder | Acoustic Encoder: EnCodec – Latent RNN Module                  | 2× LSTM                    | 2 layers                                                      |
| Semantic Encoder | Acoustic Encoder: EnCodec – Quantizer Module                   | Residual VQ                | Nq=8; codebook=1024 (10-bit)                                  |
| Semantic Decoder | Synthesizer: VALL-E – AR Transformer Module (stream 1)         | AR Transformer Module      | layers=12; d_model=1024; ffn=4096; heads=16; dropout=0.1      |
| Semantic Decoder | Synthesizer: VALL-E – NAR Transformer Modules ×7 (streams 2–8) | NAR Transformer Modules ×7 | layers=12 each; d_model=1024; ffn=4096; heads=16; dropout=0.1 |
| Semantic Decoder | Acoustic Decoder: EnCodec – 4× Upsample Module                 | Transposed Conv            | kernel=7; up-strides=8 / 5 / 4 / 2; channels=32               |


## Acknowledgments

Portions of the code in this work are referenced from the [**Amphion project**](https://github.com/open-mmlab/Amphion).  


## License


