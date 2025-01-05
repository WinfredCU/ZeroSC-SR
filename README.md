# ZeroSC-SR_public

ZeroSC-SR: Zero-Shot Semantic Communication with Speech Reconstruction

Abstract
Advancements in artificial intelligence have enabled the development of more efficient and semantic-aware speech communication systems. This repository accompanies the paper introducing an innovative Zero-Shot Semantic Communication System with Speech Reconstruction (ZeroSC-SR), designed to enhance speech transmission and reconstruction in frequency-selective fading channels.

By efficiently transmitting phoneme IDs and compressed prompt acoustic features, ZeroSC-SR achieves a high compression ratio and reduces transmitted data size, enabling superior performance under low-power and noisy conditions. We investigate the impact of prompt length on recognition accuracy and system performance, identifying a trade-off between performance and energy efficiency and offering practical guidelines for selecting appropriate prompt lengths.

Analyzing the intermediate transmission data, which includes phoneme IDs and acoustic features at different quantization levels, provides critical insights into their relative importance. These insights directly guide the development of our sort-match strategy, which enhances transmission quality in frequency-selective fading channels by allocating channel resources based on data importance.

Experimental results show that ZeroSC-SR surpasses conventional speech communication systems, demonstrating its effectiveness in real-world applications that demand reliable and efficient speech communication over challenging channels.



## Demo

A live demonstration of ZeroSC-SR is available here:  
[**ZeroSC-SR Demo**](https://winfredcu.github.io/ZeroSC-SR_demo/#abstract)

## Repository Structure

ZeroSC-SR/ ├── configs/ # Configuration files for models and experiments ├── data/ # Placeholder for data or data loading scripts ├── models/ # Model architectures and checkpoints ├── scripts/ # Auxiliary scripts (training, inference, evaluation) ├── utils/ # Utility and helper functions ├── results/ # Generated results (logs, figures, etc.) └── README.md # This README




## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WinfredCU/ZeroSC-SR.git
   cd ZeroSC-SR
   ```

2. **Install dependencies:**
   We recommend creating a new virtual environment before installing.

   ```bash
   conda create -n ZeroSC-SR python=3.8
   conda activate ZeroSC-SR
   pip install -r requirements.txt
   ```

3. **Download or prepare data:**
   - Follow the instructions in the `data/` folder to set up datasets.

## Usage

Below is a guide for running training and inference scripts. Adjust according to your specific script details.

1. **Train a Model:**
   ```bash
   python scripts/train.py \
       --config configs/your_config.yaml \
       --data_dir data/ \
       --output_dir results/train_logs
   ```

2. **Evaluate / Test a Model:**
   ```bash
   python scripts/test.py \
       --config configs/your_config.yaml \
       --checkpoint results/train_logs/checkpoint.pth \
       --output_dir results/eval_logs
   ```

3. **Speech Reconstruction Demo:**
   ```bash
   python scripts/demo.py \
       --config configs/your_demo_config.yaml \
       --input_audio path/to/input_audio.wav \
       --output_audio path/to/output_audio.wav
   ```

## Key Features

- **Zero-Shot Capability**: Works without fine-tuning on new speakers or languages.
- **High Compression Ratio**: Reduces transmission data size, beneficial in low-power and noisy conditions.
- **Frequency-Selective Fading Channels**: Employs a sort-match strategy to efficiently allocate channel resources.
- **Scalable Prompt Length**: Trade-off analysis between recognition performance and energy efficiency.

## Acknowledgments

Portions of the code in this work are referenced from the [**Amphion project**](https://github.com/open-mmlab/Amphion).  
If you use this repository and the referenced code, please cite the following:

```
@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```

## Contributing

Contributions are welcome! If you would like to make improvements or add new features, please:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/my-new-feature`).
3. Commit your changes (`git commit -am 'Add a new feature'`).
4. Push to the branch (`git push origin feature/my-new-feature`).
5. Create a pull request.

## License

This project is released under the [MIT License](LICENSE). Feel free to adapt and modify it for your own use.

---

If you have any questions or suggestions, please open an issue or submit a pull request.  
Enjoy using **ZeroSC-SR** for your speech communication and reconstruction tasks!
```
