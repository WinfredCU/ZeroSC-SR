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


## Acknowledgments

Portions of the code in this work are referenced from the [**Amphion project**](https://github.com/open-mmlab/Amphion).  


## License


