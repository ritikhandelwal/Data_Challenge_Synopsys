# Function to load audio files and extract labels
def load_audio_files(data_folders):
    X, y = [], []
    for label, folder in data_folders.items():
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                filepath = os.path.join(folder, file)
                audio, sr = librosa.load(filepath, sr=None)
                X.append(audio)
                y.append(label)
    return X, y

# Load audio files
X, y = load_audio_files(data_folders)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Confusion matrix for SVM
plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - Linear SVM")

# Confusion matrix for Random Forest
plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest")