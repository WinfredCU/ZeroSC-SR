
import numpy as np
import matplotlib.pyplot as plt
from pyldpc import make_ldpc, encode, decode


class LDPCQAMCombinedTransmission_sort:
    def __init__(self, phoneme_id_seq, acoustic_features, H, G, snr_db=15, gain_mean=0, gain_std=1, channel_type=1):
        self.phoneme_id_seq = phoneme_id_seq
        self.acoustic_features = acoustic_features
        self.snr_db = snr_db
        self.gain_mean = gain_mean
        self.gain_std = gain_std
        self.channel_type = channel_type
        self.constellation = self.generate_4qam_constellation()
        self.H = H
        self.G = G


    def index_to_binary(self, indices, bits_per_index):
        # print('indices:', indices)
        flatten_indices = indices.ravel(order='F')
        # print('flatten_indices:', flatten_indices)
        binary_strings = [format(index, '0{}b'.format(bits_per_index)) for index in flatten_indices]
        binary_array = np.array([[int(bit) for bit in string] for string in binary_strings]).flatten()
        return binary_array


    def binary_to_integers(self, binary_data, bits_per_block=8):
        num_blocks = binary_data.shape[0] // bits_per_block
        binary_blocks = binary_data[:num_blocks * bits_per_block].reshape(-1, bits_per_block)
        integers = binary_blocks.dot(2**np.arange(bits_per_block)[::-1])
        return integers


    def binary_string_to_numpy(self, binary_str):
        return np.array([int(bit) for bit in binary_str], dtype=np.int8)


    def ldpc_encode(self, binary_data):
        # Ensure the binary data length is a multiple of G's first dimension
        padding_length = (-binary_data.shape[0]) % self.G.shape[1]
        if padding_length > 0:
            binary_data = np.pad(binary_data, (0, padding_length), 'constant', constant_values=(0,))
        
        binary_data_reshaped = binary_data.reshape(-1, self.G.shape[1])
        encoded_data = np.array([encode(self.G, block, snr=1000) for block in binary_data_reshaped])
        return encoded_data.flatten()


    def ldpc_decode(self, received_data, snr):
        block_size = self.G.shape[0]
        num_blocks = received_data.shape[0] // block_size
        decoded_blocks = []
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            block = received_data[start:end]
            decoded_block = decode(self.H, block, snr)
            decoded_data = decoded_block[0:self.G.shape[1]]
            decoded_blocks.append(decoded_data)
        decoded_message = np.concatenate(decoded_blocks)
        
        # Remove any padding that was added during encoding
        original_length = self.phoneme_id_seq.size * 8 + self.acoustic_features.size * 10
        decoded_message = decoded_message[:original_length]
        
        return decoded_message


    def generate_4qam_constellation(self):
        points = np.linspace(-1, 1, 2)
        constellation = np.array([complex(i, q) for i in points for q in points])
        average_power = np.mean(np.abs(constellation)**2)
        constellation /= np.sqrt(average_power)
        return constellation


    def pad_data_for_4qam(self, ldpc_encoded_data):
        padding_length = (-ldpc_encoded_data.shape[0]) % 4
        if padding_length > 0:
            ldpc_encoded_data = np.pad(ldpc_encoded_data, (0, padding_length), 'constant', constant_values=(0,))
        return ldpc_encoded_data


    def convert_bipolar_to_binary(self, ldpc_encoded_data):
        binary_data = (ldpc_encoded_data > 0).astype(int)
        return binary_data


    def qam4_modulation(self, ldpc_encoded_data):
        ldpc_encoded_data = self.pad_data_for_4qam(ldpc_encoded_data)
        binary_data = (ldpc_encoded_data + 1) // 2
        binary_data = binary_data.reshape(-1, 2)
        indices = np.dot(binary_data, 1 << np.arange(2)[::-1])
        qam_symbols = self.constellation[indices]
        return qam_symbols


    def qam4_demodulation_and_remove_padding(self, qam_symbols, original_data_length):
        distances = np.abs(qam_symbols[:, None] - self.constellation[None, :])
        indices = np.argmin(distances, axis=1)
        binary_data = np.array([[(index >> bit) & 1 for bit in range(1, -1, -1)] for index in indices])
        binary_data = binary_data.flatten()
        ldpc_encoded_data_recovered = binary_data * 2 - 1
        ldpc_encoded_data_recovered = ldpc_encoded_data_recovered[:original_data_length]
        return ldpc_encoded_data_recovered


    def snr_to_noise_std_complex(self, signal_power):
        snr_linear = 10 ** (self.snr_db / 10)
        noise_std = np.sqrt(signal_power / (2 * snr_linear))
        return noise_std


    def channel_layer(self, qam_symbols, phoneme_length, acoustic_length):
        gain = None
        K_factor = 1
        if self.channel_type == 2:  # Rayleigh
            gain = np.sqrt(np.random.normal(self.gain_mean, self.gain_std, qam_symbols.shape)**2 +
                           np.random.normal(self.gain_mean, self.gain_std, qam_symbols.shape)**2)
        elif self.channel_type == 3:  # Rician
            rayleigh_component = np.sqrt(np.random.normal(self.gain_mean, self.gain_std, qam_symbols.shape)**2 +
                                         np.random.normal(self.gain_mean, self.gain_std, qam_symbols.shape)**2)
            deterministic_component = np.sqrt(K_factor / (K_factor + 1)) * np.ones(qam_symbols.shape)
            gain = deterministic_component + np.sqrt(1 / (2 * (K_factor + 1))) * rayleigh_component
        
        if gain is not None:
            # print('original gain:', gain)

            # Sort and match gain values to data based on importance
            gain_sorted_indices = np.argsort(-gain)
            gain_sorted = gain[gain_sorted_indices]
            # print('sorted gain:', gain_sorted)

            # qam_symbols_sorted = qam_symbols[gain_sorted_indices]
            
            # Divide gain into 9 parts: 1 for phoneme, 8 for acoustic features
            partition_sizes = [phoneme_length] + [acoustic_length // 8] * 8
            partition_indices = np.cumsum(partition_sizes)
            
            # Apply gains to respective partitions
            qam_symbols[:partition_indices[0]] *= gain_sorted[:partition_indices[0]]
            for i in range(1, 9):
                qam_symbols[partition_indices[i-1]:partition_indices[i]] *= gain_sorted[partition_indices[i-1]:partition_indices[i]]
        
        # Add noise
        signal_power = np.mean(np.abs(qam_symbols) ** 2)
        SNR_std = self.snr_to_noise_std_complex(signal_power)
        noise = np.random.normal(0, SNR_std, qam_symbols.shape) + 1j * np.random.normal(0, SNR_std, qam_symbols.shape)
        qam_symbols_noisy = qam_symbols + noise
        
        # Divide by gain to get noisy QAM symbols
        if self.channel_type in [2, 3]:
            qam_symbols_demodulated = qam_symbols_noisy / gain_sorted
        else:
            qam_symbols_demodulated = qam_symbols_noisy
        
        return qam_symbols_demodulated, gain 


    def transmit(self):
        # Step 1: Encode the phonemes and acoustic features to binary
        phoneme_binary = self.index_to_binary(self.phoneme_id_seq, 8)
        acoustic_binary = self.index_to_binary(self.acoustic_features, 10)
        # print('phoneme_binary shape:', phoneme_binary.shape)
        # print('acoustic_binary shape:', acoustic_binary.shape)

        combined_binary = np.concatenate((phoneme_binary, acoustic_binary))
        # print('combined_binary shape:', combined_binary.shape)

        # print('phoneme_binary:', phoneme_binary)
        # print('acoustic_binary:', acoustic_binary)
        # print('combined_binary:', combined_binary)
       
        # Step 2: LDPC Encoding
        ldpc_encoded_data = self.ldpc_encode(combined_binary)
        # print('ldpc_encoded_data shape:', ldpc_encoded_data.shape)

        # Step 3: QAM Modulation
        qam_symbols = self.qam4_modulation(self.convert_bipolar_to_binary(ldpc_encoded_data))
        # print('qam_symbols shape:', qam_symbols.shape)
        
        # Step 4: Transmit through a channel
        # encoded_phoneme_length = len(phoneme_binary) * (self.G.shape[0] // self.G.shape[1])
        # encoded_acoustic_length = len(acoustic_binary) * (self.G.shape[0] // self.G.shape[1])

        encoded_phoneme_length = self.G.shape[0] * (len(phoneme_binary) // self.G.shape[1] + 1)
        encoded_acoustic_length = self.G.shape[0]  * (len(acoustic_binary) // self.G.shape[1])
        phoneme_length = encoded_phoneme_length // 2  # Each QAM symbol represents 2 bits
        acoustic_length = encoded_acoustic_length // 2  # Each QAM symbol represents 2 bits

        # print('phoneme_length:', phoneme_length)
        # print('acoustic_length:', acoustic_length)

        qam_symbols_noisy, gain = self.channel_layer(qam_symbols, phoneme_length, acoustic_length)

        # Step 5: QAM Demodulation
        original_data_length = ldpc_encoded_data.shape[0]
        QAM_demodulated = self.qam4_demodulation_and_remove_padding(qam_symbols_noisy, original_data_length)
        
        # Step 6: LDPC Decoding
        snr = 1000
        decoded_data = self.ldpc_decode(QAM_demodulated, snr)
        
        # Step 7: Split decoded data back into phonemes and acoustic features
        phoneme_length_bits = self.phoneme_id_seq.size * 8
        acoustic_length_bits = self.acoustic_features.size * 10
        phoneme_data = decoded_data[:phoneme_length_bits]
        acoustic_data = decoded_data[phoneme_length_bits:phoneme_length_bits + acoustic_length_bits]
        
        # Step 8: Convert binary data back to integers
        phoneme_output = self.binary_to_integers(phoneme_data, 8)
        acoustic_output = self.binary_to_integers(acoustic_data, 10)
        
        return phoneme_output, acoustic_output


import matplotlib.pyplot as plt

# Test code
if __name__ == "__main__":
    # Generate LDPC matrices
    n, d_v, d_c = 16, 5, 8
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

    print('G shape:',G.shape)

    # Sample input data
    input_phoneme_matrix = np.random.randint(0, 256, size=(200, 1))
    input_acoustic_matrix = np.random.randint(0, 1024, size=(200, 8))

    # Set SNR
    snr_db = 18

    # Initialize the transmitter
    transmitter = LDPCQAMCombinedTransmission_sort(input_phoneme_matrix, input_acoustic_matrix, H, G, snr_db=snr_db, gain_mean=0, gain_std=1, channel_type=2)

    # Transmit data
    transmitted_output_phoneme, transmitted_output_acoustic = transmitter.transmit()

    # # Plot gain distribution
    # _, gain = transmitter.channel_layer(transmitter.qam4_modulation(transmitter.convert_bipolar_to_binary(transmitter.ldpc_encode(transmitter.index_to_binary(input_phoneme_matrix, 8)))))
    # plt.hist(gain, bins=50, alpha=0.7, color='blue')
    # plt.title('Gain Distribution')
    # plt.xlabel('Gain Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    original_shape = input_acoustic_matrix.shape

    # Flatten input data for comparison
    input_phoneme_data = input_phoneme_matrix.flatten()
    input_acoustic_data = input_acoustic_matrix.flatten()

    # Function to compare input and output
    def compare_and_report(array1, array2):
        if array1.shape != array2.shape:
            return "Arrays have different shapes."
        differences = np.where(array1 != array2)
        if differences[0].shape[0] > 0:
            return f"Differences at positions {differences}"
        else:
            return "Arrays are identical."

    # Compare input and output
    print("Comparison report for phoneme data:", compare_and_report(input_phoneme_data, transmitted_output_phoneme))

    print('input_acoustic_data:', input_acoustic_matrix)

    reshaped_output = transmitted_output_acoustic.reshape(original_shape, order='F')
    print('transmitted_output_acoustic:', reshaped_output)


    print("Comparison report for acoustic data:", compare_and_report(input_acoustic_matrix.flatten(), reshaped_output.flatten()))
