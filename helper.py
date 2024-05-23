import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import librosa.display

def clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all the files and directories inside the specified directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The directory {directory_path} does not exist.')

def initialize_csv(csv_path, n_mfcc=10, n_chroma=12, n_spc=7):
    if not os.path.exists(csv_path):
        headers = ["filename", "zero_crossing", "mean_sprectral_roll_off"] + [f"mfcc_{i+1}" for i in range(n_mfcc)] + [f"chroma_{i+1}" for i in range(n_chroma)] + [f"spectral_contrast_{i+1}" for i in range(n_spc)]
        df = pd.DataFrame(columns=headers)
        # df.set_index("filename", inplace=True)
        df.to_csv(csv_path, index=False)


def update_csv(csv_path, filename, array, type):
    df = pd.read_csv(csv_path)
    for i, value in enumerate(array):
        df.loc[df['filename'] == filename,  f'{type}_{i+1}'] = float(value)
    df.to_csv(csv_path, index=False)

def plot_func(subdir, filename, type, S, **kwargs):
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(S, **kwargs)
    file_path = os.path.join(subdir, f'{filename[:-4]}_{type}.png')
    plt.savefig(file_path)
    plt.clf()