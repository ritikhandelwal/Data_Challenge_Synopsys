import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# # Paths to data folders
# data_folders = {
#     'air_conditioner': 'E:/Synopsys_Data_Challenge/data/air_conditioner',
#     'car_horn': 'E:/Synopsys_Data_Challenge/data/car_horn',
#     'engine_idling': 'E:/Synopsys_Data_Challenge/data/engine_idling',
#     'siren': 'E:/Synopsys_Data_Challenge/data/siren'
# }

# Feature extraction function
def extract_features(audio, sr):
    n_fft = min(2048, len(audio) // 2)
    n_mels = max(13, n_fft // 2 // 10)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, n_mels=n_mels)
    mfccs = np.mean(mfccs.T, axis=0)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))                           
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    return np.hstack([mfccs, zcr, spectral_centroid])
