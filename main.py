#import kagglehub
from glob import glob
import librosa
import librosa.display
import numpy as np
import matplotlib as matplotlib


#path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
#path = "/Users/achintya.san/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1"
#print("Path to dataset files:", path)

test_audio_file = "archive-2/Actor_01/03-01-01-01-01-01-01.wav"

audio, sample_rate = librosa.load(test_audio_file)

FRAME_SIZE = 2048
HOP_SIZE = 512
S_audio = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print(S_audio.shape)
Y_audio = np.abs(S_audio) ** 2
print(Y_audio.shape)

def plot_spectogram(Y, sr, hop_length, y_axis="linear"):
    matplotlib.pyplot.figure(figsize=(25,10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    matplotlib.pyplot.colorbar(format="%+2.f")

#plot_spectogram(Y_audio, sample_rate, HOP_SIZE)

Y_log_audio = librosa.power_to_db(Y_audio)
plot_spectogram(Y_log_audio, sample_rate, HOP_SIZE, y_axis="log")




#Sources:
#https://www.youtube.com/watch?v=3gzI4Z2OFgY&t=294s
#https://www.youtube.com/watch?v=ZqpSb5p1xQo
