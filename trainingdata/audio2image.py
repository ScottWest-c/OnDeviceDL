import kagglehub
from glob import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from PIL import Image

#path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
#path = "/Users/achintya.san/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1"
#print("Path to dataset files:", path)

#test_audio_file = "/Users/achintya.san/Desktop/CS556FinalProject/trainingdata/archive-2/Actor_16/03-01-05-01-02-01-16.wav"
# test_audio_file = "archive-2/Actor_01/03-01-01-01-01-01-01.wav"



train_label_1 = Path.cwd() / "archive-3/clean_trainset_28spk_wav"
train_label_2 = Path.cwd() / "archive-3/clean_trainset_56spk_wav"

train_data_1= Path.cwd() / "archive-3/noisy_trainset_28spk_wav"
train_data_2 = Path.cwd() / "archive-3/noisy_trainset_56spk_wav"

#files = list(input_dir.rglob("*.wav*"))
#files = ["archive-3/noisy_trainset_56spk_wav/p226_001.wav"]
#files[0] = "archive-3/noisy_trainset_56spk_wav/p226_001.wav"

i = 1
#for file in files:
#audio_file = file
audio_file = "archive-3/noisy_trainset_28spk_wav/p226_001.wav"    #.02*sample_rate 

audio, sample_rate = librosa.load(audio_file)
#print(sample_rate)

FRAME_SIZE = 2048
HOP_SIZE = 512
S_audio = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_audio = np.abs(S_audio) ** 2


Y_log_audio = librosa.power_to_db(Y_audio)

plt.figure(figsize=(10,6))
librosa.display.specshow(librosa.amplitude_to_db(Y_log_audio, ref=np.max), sr=sample_rate, hop_length=HOP_SIZE)
# plt.colorbar(format='%+2.0f dB')
# plt.title("Spectogram")
plt.axis("off")
plt.gca().set_xticks([])  # Remove x ticks
plt.gca().set_yticks([])  # Remove y ticks
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove extra padding
plt.savefig(f"spectogram_images/spectogram_{i}.png")
plt.close()  
i = i + 1
#print(Y_log_audio.shape)
#plot_spectogram(Y_log_audio, sample_rate, HOP_SIZE, y_axis="log")





 


#Sources:
#https://www.youtube.com/watch?v=3gzI4Z2OFgY&t=294s
#https://www.youtube.com/watch?v=ZqpSb5p1xQo
#https://www.youtube.com/watch?v=w6-28jcr09Q