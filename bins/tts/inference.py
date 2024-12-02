import argparse
from argparse import ArgumentParser
import os


from models.tts.valle.valle_inference_SortMatch import VALLEInference

from utils.util import load_config
import torch
import numpy

def build_inference(args, cfg):
    supported_inference = {
        "VALLE": VALLEInference,
    }
    inference_class = supported_inference[cfg.model_type]
    inference = inference_class(args, cfg)
    return inference


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON/YAML file for configurations.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="convert from the source data",
        default=None,
    )
    parser.add_argument(
        "--testing_set",
        type=str,
        help="train, test, golden_test",
        default="test",
    )
    parser.add_argument(
        "--test_list_file",
        type=str,
        help="convert from the test list file",
        default=None,
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default=None,
        help="speaker name for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--text",
        help="Text to be synthesized.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--vocoder_dir",
        type=str,
        default=None,
        help="Vocoder checkpoint directory. Searching behavior is the same as "
        "the acoustics one.",
    )
    parser.add_argument(
        "--acoustics_dir",
        type=str,
        default=None,
        help="Acoustic model checkpoint directory. If a directory is given, "
        "search for the latest checkpoint dir in the directory. If a specific "
        "checkpoint dir is given, directly load the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Acoustic model checkpoint directory. If a directory is given, "
        "search for the latest checkpoint dir in the directory. If a specific "
        "checkpoint dir is given, directly load the checkpoint.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        help="Logging level. Default: warning",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output dir for saving generated results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='Experiment_Data/output/test.wav',
        help="Output path for saving the generated audio",
    )
    parser.add_argument(
        "--strategy_mode",
        type=str,
        default='sort',
        help='The strategy modes: clean, sort, random'
    )
    parser.add_argument(
        "--SNR",
        type=float,
        default=3.0,
        help='dB'
    )
    parser.add_argument(
        "--CodecStage",
        type=float,
        default='1.0'
    )
    parser.add_argument(
        "--DropRate",
        type=float,
        default='0.8'
    )
    parser.add_argument(
        "--power",
        type=float,
        default='10'
    )

    return parser


def main():
    # Parse arguments
    parser = build_parser()
    VALLEInference.add_arguments(parser)
    args = parser.parse_args()
    print(args)

    # Parse config
    cfg = load_config(args.config)

    # CUDA settings
    cuda_relevant()

    # Build inference
    inferencer = build_inference(args, cfg)

    # Run inference
    inferencer.inference()


if __name__ == "__main__":
    main()
