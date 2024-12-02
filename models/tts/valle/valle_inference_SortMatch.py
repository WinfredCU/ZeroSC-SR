# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import torchaudio
import argparse


from text.g2p_module import G2PModule
from utils.tokenizer import AudioTokenizer, tokenize_audio
from models.tts.valle.valle import VALLE
from models.tts.base.tts_inferece import TTSInference
from models.tts.valle.valle_dataset import VALLETestDataset, VALLETestCollator
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation


# from VALLE.Scripts_VALLE.Transmission.CombineTransmitClass_sortmatch import LDPCQAMCombinedTransmission_sort
# from VALLE.Scripts_VALLE.Transmission.CombineTransmitClass_randommatch import LDPCQAMCombinedTransmission_random

from Transmission.CombineTransmitClass_sortmatch import LDPCQAMCombinedTransmission_sort
from Transmission.CombineTransmitClass_randommatch import LDPCQAMCombinedTransmission_random



class VALLEInference(TTSInference):
    def __init__(self, args=None, cfg=None):
        TTSInference.__init__(self, args, cfg)

        self.g2p_module = G2PModule(backend=self.cfg.preprocess.phone_extractor)
        text_token_path = os.path.join(
            cfg.preprocess.processed_dir, cfg.dataset[0], cfg.preprocess.symbols_dict
        )
        self.audio_tokenizer = AudioTokenizer()

    def _build_model(self):
        model = VALLE(self.cfg.model)
        return model

    def _build_test_dataset(self):
        return VALLETestDataset, VALLETestCollator

    def inference_one_clip(self, text, text_prompt, audio_file, save_name="pred"):
        # get phone symbol file
        phone_symbol_file = None
        if self.cfg.preprocess.phone_extractor != "lexicon":
            phone_symbol_file = os.path.join(
                self.exp_dir, self.cfg.preprocess.symbols_dict
            )
            # print('phone_symbol_file:',phone_symbol_file)
            assert os.path.exists(phone_symbol_file)
        # convert text to phone sequence
        phone_extractor = phoneExtractor(self.cfg)
        # convert phone sequence to phone id sequence
        phon_id_collator = phoneIDCollation(
            self.cfg, symbols_dict_file=phone_symbol_file
        )

        text = f"{text_prompt} {text}".strip()
        phone_seq = phone_extractor.extract_phone(text)  # phone_seq: list
        phone_id_seq = phon_id_collator.get_phone_id_sequence(self.cfg, phone_seq)
        phone_id_seq_len = torch.IntTensor([len(phone_id_seq)]).to(self.device)

        # extract acoustic token
        encoded_frames = tokenize_audio(self.audio_tokenizer, audio_file)

        mode = self.args.strategy_mode

        print('strategy mode:', mode)

        snr_db = self.args.SNR
        print('SNR:', snr_db)

        print('--------------------Transmission Begin--------------------------')

        # # ####################################### 3. Clean Code  #######################################
            
        if mode == "clean":

            phone_id_seq = np.array([phone_id_seq])
            phone_id_seq = torch.from_numpy(phone_id_seq).to(self.device)
            audio_prompt_token = encoded_frames[0][0].transpose(2, 1).to(self.device)

        elif mode == "sort":

            # H, G generation 
            H_Audio_Path = 'H.npy'
            G_Audio_Path = 'G.npy'

            with open(H_Audio_Path, 'rb') as f:
                H = np.load(f)
            with open(G_Audio_Path, 'rb') as f:
                G = np.load(f)

            ########### Phoneme Sequence Input ################
            # convert phone sequence to phone id sequence
            phone_id_seq = np.array([phone_id_seq])

            # print('phone_id_seq shape:', phone_id_seq.shape)

            ########### Audio Input ##########################
            audio_prompt_token1 = encoded_frames[0][0].transpose(2, 1)

            squeezed_tensor = audio_prompt_token1.squeeze(0)
            # Make sure to transfer the tensor to the CPU before converting to a NumPy array.
            audio_prompts_numpy = squeezed_tensor.cpu().numpy()
            original_shape = audio_prompts_numpy.shape


            ################################ Transmission ###########################
            # Initialize the transmitter
            transmitter = LDPCQAMCombinedTransmission_sort(phone_id_seq, audio_prompts_numpy, H, G, snr_db=snr_db, gain_mean=0, gain_std=1, channel_type=2)

            # Transmit data
            transmitted_output_phoneme, transmitted_output_acoustic = transmitter.transmit()

            ################## Phoneme Output ########################

            transmitted_output_phoneme = np.expand_dims(transmitted_output_phoneme, axis=0)
            # print('transmitted_output_phoneme shape:', transmitted_output_phoneme.shape)
            # print('transmitted_output_phoneme:', transmitted_output_phoneme)
            phone_id_seq = torch.from_numpy(transmitted_output_phoneme).to(self.device)

            print("Success: Phoneme Transmission !!!")

            ################# Audio Output #########################

            reshaped_output = transmitted_output_acoustic.reshape(original_shape, order='F')

            processed_data2 = np.concatenate([arr.reshape(1, -1) for arr in reshaped_output], axis=0)
            tensor_data2 = torch.tensor(processed_data2, dtype=torch.int32)
            # Reshape tensor
            input_tensor2 = tensor_data2.unsqueeze(0)  # Adds a batch dimension
            audio_prompt_token = input_tensor2.to(self.device)

            # print("audio prompt:")
            # print(audio_prompt_token)
            # print(audio_prompt_token.shape)  #torch.Size([1, 226, 8])

            print("Success: Audio Codec Transmission !!!")

        elif mode == "random":

            # H, G generation 
            H_Audio_Path = 'H.npy'
            G_Audio_Path = 'G.npy'

            with open(H_Audio_Path, 'rb') as f:
                H = np.load(f)
            with open(G_Audio_Path, 'rb') as f:
                G = np.load(f)

            ########### Phoneme Sequence Input ################
            # convert phone sequence to phone id sequence
            phone_id_seq = np.array([phone_id_seq])

            # print('phone_id_seq shape:', phone_id_seq.shape)

            ########### Audio Input ##########################
            audio_prompt_token1 = encoded_frames[0][0].transpose(2, 1)

            squeezed_tensor = audio_prompt_token1.squeeze(0)
            # Make sure to transfer the tensor to the CPU before converting to a NumPy array.
            audio_prompts_numpy = squeezed_tensor.cpu().numpy()
            original_shape = audio_prompts_numpy.shape


            ################################ Transmission ###########################
            # Initialize the transmitter
            transmitter = LDPCQAMCombinedTransmission_random(phone_id_seq, audio_prompts_numpy, H, G, snr_db=snr_db, gain_mean=0, gain_std=1, channel_type=2)

            # Transmit data
            transmitted_output_phoneme, transmitted_output_acoustic = transmitter.transmit()

            ################## Phoneme Output ########################

            transmitted_output_phoneme = np.expand_dims(transmitted_output_phoneme, axis=0)
            # print('transmitted_output_phoneme shape:', transmitted_output_phoneme.shape)
            # print('transmitted_output_phoneme:', transmitted_output_phoneme)
            phone_id_seq = torch.from_numpy(transmitted_output_phoneme).to(self.device)

            print("Success: Phoneme Transmission !!!")

            ################# Audio Output #########################

            reshaped_output = transmitted_output_acoustic.reshape(original_shape, order='F')

            processed_data2 = np.concatenate([arr.reshape(1, -1) for arr in reshaped_output], axis=0)
            tensor_data2 = torch.tensor(processed_data2, dtype=torch.int32)
            # Reshape tensor
            input_tensor2 = tensor_data2.unsqueeze(0)  # Adds a batch dimension
            audio_prompt_token = input_tensor2.to(self.device)

            print("audio prompt:")
            # print(audio_prompt_token)
            print(audio_prompt_token.shape)  #torch.Size([1, 226, 8])

            print("Success: Audio Codec Transmission !!!")

        #############################################################################################
        # copysyn
        if self.args.copysyn:
            samples = self.audio_tokenizer.decode(encoded_frames)
            audio_copysyn = samples[0].cpu().detach()

            out_path = os.path.join(
                self.args.output_dir, self.infer_type, f"{save_name}_copysyn.wav"
            )
            torchaudio.save(out_path, audio_copysyn, self.cfg.preprocess.sampling_rate)

        if self.args.continual:
            # print('phone_id_seq:',phone_id_seq)
            # print('phone_id_seq_len:',phone_id_seq_len)
            # print('audio_prompt_token:',audio_prompt_token)
            encoded_frames = self.model.continual(
                phone_id_seq,
                phone_id_seq_len,
                audio_prompt_token,
            )
            print("continual True")
            # print('encoded_frames:',encoded_frames)
            print(encoded_frames.shape)
        else:
            enroll_x_lens = None
            print("continual False")

            print(text_prompt)

            if text_prompt:
                # prompt_phone_seq = tokenize_text(self.g2p_module, text=f"{text_prompt}".strip())
                # _, enroll_x_lens = self.text_tokenizer.get_token_id_seq(prompt_phone_seq)

                text = f"{text_prompt}".strip()
                prompt_phone_seq = phone_extractor.extract_phone(
                    text
                )  # phone_seq: list
                prompt_phone_id_seq = phon_id_collator.get_phone_id_sequence(
                    self.cfg, prompt_phone_seq
                )
                prompt_phone_id_seq_len = torch.IntTensor(
                    [len(prompt_phone_id_seq)]
                ).to(self.device)


            encoded_frames = self.model.inference(
                phone_id_seq,
                phone_id_seq_len,
                audio_prompt_token,
                enroll_x_lens=prompt_phone_id_seq_len,
                top_k=self.args.top_k,
                temperature=self.args.temperature,
            )
            print(encoded_frames.shape)


        samples = self.audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])

        audio = samples[0].squeeze(0).cpu().detach()

        return audio

    def inference_for_single_utterance(self):
        text = self.args.text
        text_prompt = self.args.text_prompt
        audio_file = self.args.audio_prompt

        if not self.args.continual:
            assert text != ""
        else:
            text = ""
        assert text_prompt != ""
        assert audio_file != ""

        audio = self.inference_one_clip(text, text_prompt, audio_file)

        return audio

    def inference_for_batches(self):
        test_list_file = self.args.test_list_file
        assert test_list_file is not None

        pred_res = []
        with open(test_list_file, "r") as fin:
            for idx, line in enumerate(fin.readlines()):
                fields = line.strip().split("|")
                if self.args.continual:
                    assert len(fields) == 2
                    text_prompt, audio_prompt_path = fields
                    text = ""
                else:
                    assert len(fields) == 3
                    text_prompt, audio_prompt_path, text = fields

                audio = self.inference_one_clip(
                    text, text_prompt, audio_prompt_path, str(idx)
                )
                pred_res.append(audio)

        return pred_res


    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--text_prompt",
            type=str,
            default="",
            help="Text prompt that should be aligned with --audio_prompt.",
        )

        parser.add_argument(
            "--audio_prompt",
            type=str,
            default="",
            help="Audio prompt that should be aligned with --text_prompt.",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=-100,
            help="Whether AR Decoder do top_k(if > 0) sampling.",
        )

        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="The temperature of AR Decoder top_k sampling.",
        )

        parser.add_argument(
            "--continual",
            action="store_true",
            default=False,
            help="Inference for continual task.",
        )

        parser.add_argument(
            "--copysyn",
            action="store_true",
            help="Copysyn: generate audio by decoder of the original audio tokenizer.",
        )
