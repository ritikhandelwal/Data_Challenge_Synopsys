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

# Extract features for training and testing sets
X_train_features = np.array([extract_features(audio, sr=44100) for audio in X_train])
X_test_features = np.array([extract_features(audio, sr=44100) for audio in X_test])

# Standardize the features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

# Train a linear SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_features, y_train)

# Predict on test set
y_pred_svm = svm_model.predict(X_test_features)

# Evaluate the model
print("Linear SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Train a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_features, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test_features)

# Evaluate the model
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
