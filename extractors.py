import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import scale
import numpy as np
from helper import *


def plot_waveform(subdir, filename, y, sr):
    plt.figure(figsize=(15, 4), facecolor=(.5, .5, .5))
    librosa.display.waveshow(y, sr=sr, color='pink')

    file_path = os.path.join(subdir, filename[:-4]+"_waveform_plot.png")
    plt.savefig(file_path)
    plt.clf()
     

def plot_spectogram(subdir, filename, y, sr):
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    file_path = os.path.join(subdir, filename[:-4]+"_spectrum.png")
    plt.savefig(file_path)
    plt.clf()




def calc_mfcc(subdir, filename, y, sr, csv_path):
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = scale(mfccs, axis=1)
    plot_func(subdir, filename, type='mfcc', S=mfccs, sr=sr, x_axis='time')
    mfccs_mean = np.mean(mfccs, axis=1)
    update_csv(csv_path, filename, mfccs_mean,'mfcc')



def calc_chroma(subdir, filename, y, sr, csv_path, hop_length=512):
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    plot_func(subdir, filename, type='chroma', S=chromagram,x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    chroma_mean = np.mean(chromagram, axis=1) 
    update_csv(csv_path, filename, chroma_mean, 'chroma')




def calc_spectral_contrast(subdir, filename, y, sr, csv_path):
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    plot_func(subdir, filename, type='spectral_contrast', S=contrast, sr=sr, x_axis='time')
    contrast_mean = np.mean(contrast, axis=1)
    update_csv(csv_path, filename, contrast_mean,'spectral_contrast')

def count_zero_crossing(subdir, filename, y, sr, csv_path):
    zero_crossings = librosa.zero_crossings(y,pad=False)
    df = pd.read_csv(csv_path)
    df.loc[df['filename'] == filename,  'zero_crossing'] = float(sum(zero_crossings))
    df.to_csv(csv_path, index=False)

def calc_spectral_roll_off(subdir,filename, y, sr, csv_path):
    rool_off = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    df = pd.read_csv(csv_path)
    df.loc[df['filename'] == filename,  'zero_crossing'] = float(np.mean(rool_off))
    df.to_csv(csv_path, index=False)