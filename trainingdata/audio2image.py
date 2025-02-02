#import kagglehub
from glob import glob
import librosa
import librosa.display
import numpy as np
import matplotlib as matplotlib
import pandas as pd
from pathlib import Path
import os
from PIL import Image

#path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
#path = "/Users/achintya.san/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1"
#print("Path to dataset files:", path)

#test_audio_file = "/Users/achintya.san/Desktop/CS556FinalProject/trainingdata/archive-2/Actor_16/03-01-05-01-02-01-16.wav"
# test_audio_file = "archive-2/Actor_01/03-01-01-01-01-01-01.wav"

input_dir = Path.cwd() / "archive-2"
files = list(input_dir.rglob("*.wav*"))
# spectograms = []
i = 0
for file in files:
    audio_file = file

    audio, sample_rate = librosa.load(audio_file)

    FRAME_SIZE = 2048
    HOP_SIZE = 512
    S_audio = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    #print(S_audio.shape)
    Y_audio = np.abs(S_audio) ** 2
    #print(Y_audio.shape)

    def plot_spectogram(Y, sr, hop_length, y_axis="linear"):
        matplotlib.pyplot.figure(figsize=(25,10))
        librosa.display.specshow(Y,
                                sr=sr,
                                hop_length=hop_length,
                                x_axis="time",
                                y_axis=y_axis)
        matplotlib.pyplot.colorbar(format="%+2.f")
        #matplotlib.pyplot.close()

    #plot_spectogram(Y_audio, sample_rate, HOP_SIZE)

    Y_log_audio = librosa.power_to_db(Y_audio)
    #print(Y_log_audio.shape)
    #plot_spectogram(Y_log_audio, sample_rate, HOP_SIZE, y_axis="log")

    if not isinstance(Y_log_audio, Image.Image):
        Y_log_audio = Image.fromarray(Y_log_audio)
        if Y_log_audio.mode != 'RGB':
            Y_log_audio = Y_log_audio.convert('RGB')

    filename = os.path.join("spectogram_images", f"spectogram_{i}.jpg")
    Y_log_audio.save(filename)
    i = i + 1

    # spectograms.append(Y_log_audio)

# for i, spectogram in enumerate(spectograms):
#     if not isinstance(spectogram, Image.Image):
#         spectogram = Image.fromarray(spectogram)
#         if spectogram.mode != 'RGB':
#             spectogram = spectogram.convert('RGB')

#     filename = os.path.join("spectogram_images", f"spectogram_{i}.jpg")

#     spectogram.save(filename)

 


#Sources:
#https://www.youtube.com/watch?v=3gzI4Z2OFgY&t=294s
#https://www.youtube.com/watch?v=ZqpSb5p1xQo
#https://www.youtube.com/watch?v=w6-28jcr09Q