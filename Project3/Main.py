#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:37:23 2022

@author: karlykussainova
"""

from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play   
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import librosa
import os
from PIL import Image
import pathlib
#import csv from sklearn.model_selection 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import tree

# import keras
# from keras import layers
# from keras import layers
# import keras
# from keras.models import Sequential
# import warnings
# warnings.filterwarnings('ignore')


 
# Get the list of all files and directories
path = "Audios/wav"
dir_list = os.listdir(path)

files_lst = []
label_lst = []
for i in range(len(dir_list)):
    if '.wav' in dir_list[i]:
        files_lst.append(dir_list[i])
        if 'neg' in dir_list[i]:
            label_lst.append('negative')
        if 'neutral' in dir_list[i]:
            label_lst.append('neutral')
        if 'pos' in dir_list[i]:
            label_lst.append('positive') 

path_to_wav = "audios/wav/"
channels_lst = []
sample_width_lst = []
rmse_lst = []
chroma_stft_lst = []
spec_cent_lst = []
spec_bw_lst = []
rolloff_lst = []
zcr_lst = []
mfcc_lst = []

for i in range(len(files_lst)):
    file = path_to_wav + files_lst[i]
    audio_segment = AudioSegment.from_file(file)
    channels_lst.append(audio_segment.channels)
    sample_width_lst.append(audio_segment.sample_width)
    
    y, sr = librosa.load('audios/wav/neg4.wav', mono=True, duration=30)
    rmse = librosa.feature.rms(y=y)
    rmse_lst.append(rmse)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_lst.append(chroma_stft)
    
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_cent_lst.append(spec_cent)
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw_lst.append(spec_bw)
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_lst.append(rolloff)
    
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_lst.append(zcr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_lst.append(mfcc)
    
    
# dictionary of lists 
dict = {'file_name': files_lst, 'channels': channels_lst, 'sample_width': sample_width_lst,
        'rmse' : rmse_lst,   'chroma_stft': chroma_stft_lst, 'spec_cent': spec_cent_lst,
        'spec_bw' : spec_bw_lst, 'rolloff': rolloff_lst, 'zcr': zcr_lst, 'mfcc': mfcc_lst,
        'label' : label_lst} 
    
df = pd.DataFrame(dict)


data = df.drop(['file_name'],axis=1)#Encoding the Labels
labels = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(labels)#Scaling the Feature columns
scaler = StandardScaler()
#X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
X = data.iloc[:, 0:9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

result = clf.predict(X_test)
print(result)