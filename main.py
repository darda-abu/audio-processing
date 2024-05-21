import librosa
import os

main_directory = 'data'
plot_directory = 'audio_plots'

for filename in os.listdir(main_directory):
    if filename.endswith(".wav"):
        print(filename)
        y, sr = librosa.load(os.path.join(main_directory, filename))
        


%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display