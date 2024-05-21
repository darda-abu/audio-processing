import librosa
import os

import matplotlib.pyplot as plt
import librosa.display


main_directory = 'data//audio_files'
plot_directory = 'audio_plots'
if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)



def plot_waveform(filename):
     


for filename in os.listdir(main_directory):
    if filename.endswith(".wav"):
        print(filename)
    y, sr = librosa.load(os.path.join(main_directory, filename))
        



    plt.figure(figsize=(15, 4), facecolor=(.5, .5, .5))
    librosa.display.waveshow(y, sr=sr, color='pink')

    file_path = os.path.join(plot_directory, filename[:-4]+"_waveform_plot.png")
    plt.savefig(file_path)
    plt.clf()

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))