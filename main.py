import librosa
import os
import pandas as pd
import librosa.display
import warnings
from helper import *
from extractors import *
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

main_directory = 'data//audio_files'
plot_directory = 'audio_plots'
csv_path = "audio_features.csv"


if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

clear_directory(plot_directory)

if os.path.exists(csv_path): os.remove(csv_path)
initialize_csv(csv_path)
    


for filename in os.listdir(main_directory):
    if filename.endswith(".wav"):
        print("processing", filename)

    y, sr = librosa.load(os.path.join(main_directory, filename))
        
    subdir = os.path.join(plot_directory, filename[:-4])

    if not os.path.exists(subdir):
        os.makedirs(subdir)
        
    df = pd.read_csv(csv_path)
    if filename not in df['filename'].values:
        df.loc[len(df),'filename'] = filename
    df.to_csv(csv_path, index=False)
    
    plot_waveform(subdir, filename, y, sr)
    plot_spectogram(subdir, filename, y, sr)
    calc_mfcc(subdir, filename, y, sr, csv_path)
    calc_chroma(subdir, filename, y, sr, csv_path)
    calc_spectral_contrast(subdir, filename, y, sr, csv_path)
    count_zero_crossing(subdir, filename, y, sr, csv_path)
    calc_spectral_roll_off(subdir, filename, y, sr, csv_path)
