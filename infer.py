import os
import subprocess


# Function to generate output path from prompt audio path
def get_output_path(prompt_audio_path, output_folder):
    # Extract the relative part of the path after the prompt folder structure
    relative_path = os.path.relpath(prompt_audio_path, start=prompt_text_folder)
    
    # Break down the file name to remove the part after the last underscore before the extension
    dir_name, base_name = os.path.split(relative_path)
    name_part, extension = os.path.splitext(base_name)
    name_parts = name_part.split('_')
    new_name_part = '_'.join(name_parts[:-1])  # Remove the last part after the last underscore
    new_base_name = f"{new_name_part}{extension}"  # Reassemble the file name without the last part
    
    # Reconstruct the full path with the modified file name
    return os.path.join(output_folder, dir_name, new_base_name)


def process_files(prompt_text_folder, target_text_folder, output_folder, limit=None):
    all_files = []
    
    # Gather all the .txt files first
    for root, dirs, files in os.walk(prompt_text_folder):
        for filename in files:
            if filename.endswith('.txt'):
                all_files.append(os.path.join(root, filename))
    
    total_files = len(all_files)
    
    for index, prompt_text_path in enumerate(all_files):
        if limit and index >= limit:
            break
        
        prompt_audio_path = prompt_text_path.replace('.txt', '.wav')  # Assuming audio file matches text file name
        target_text_path = find_target_text_path(target_text_folder, prompt_text_path)

        print("--"*60)

        print(f"Processing file {index + 1}/{total_files}")
        print(f"Prompt text path: {prompt_text_path}")
        print(f"Prompt audio path: {prompt_audio_path}")
        print(f"Target text path: {target_text_path}")

        if not os.path.exists(target_text_path):
            print(f"Target text file not found for {target_text_path}")
            continue

        with open(target_text_path, 'r', encoding='utf-8') as file:
            target_text = file.read()

        with open(prompt_text_path, 'r', encoding='utf-8') as file:
            prompt_text = file.read()

        print(target_text)
        print(prompt_text)

        # Generate the output path
        output_path = get_output_path(prompt_audio_path, output_folder)
        print(output_path)

        command = [
            'bash', 'egs/tts/VALLE/run.sh', '--stage', '3', '--gpu', '0',
            '--config', 'ckpts/tts/valle_librilight_6k/args.json',
            '--infer_expt_dir', 'ckpts/tts/valle_librilight_6k',
            '--infer_output_dir', output_folder,
            '--infer_mode', 'single',
            '--infer_text', target_text,
            '--infer_text_prompt', prompt_text,
            '--infer_audio_prompt', prompt_audio_path,
            '--infer_output_path', output_path,
            '--infer_strategy_mode', mode,
            '--infer_SNR', '{}'.format(snr_db)

        ]
        subprocess.run(command)

# Function to find the corresponding target text path based on the prompt text path
def find_target_text_path(target_text_folder, prompt_text_path):
    prompt_basename = os.path.basename(prompt_text_path)
    # print(prompt_basename)
    target_basename = prompt_basename.replace('_2s.txt', '.recognized.txt') 
    return os.path.join(target_text_folder, target_basename)  # Reconstruct the path with the new base name


# # Define the directories
prompt_text_folder = 'Experiment_Data/prompt/length/2s'
target_text_folder = 'Experiment_Data/raw/test-clean'


snr_db_list = [5, 10, 0, 8]
mode = 'sort' # 'clean' 'random'

for snr_db in snr_db_list:

    output_folder='Experiment_Data/output/{}dB_{}'.format(snr_db, mode)
    print(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    # Example usage:
    process_files(prompt_text_folder,target_text_folder, output_folder, limit=1000)
