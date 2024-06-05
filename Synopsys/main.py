import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, f'{title}.png'))
    plt.show()

def extract_features(audio, sr):
    n_fft = min(2048, len(audio) // 2)
    n_mels = max(13, n_fft // 2 // 10)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, n_mels=n_mels)
    mfccs = np.mean(mfccs.T, axis=0)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))                           
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    return np.hstack([mfccs, zcr, spectral_centroid])

def run_job(X, y, class_names, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    print(classification_report(y_test, y_pred_svm, target_names=class_names))

    # Train a Random Forest classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_features, y_train)

    # Predict on test set
    y_pred_rf = rf_model.predict(X_test_features)

    # Evaluate the model
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))

    # Plot and save confusion matrices
    plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - Linear SVM", class_names, output_dir)
    plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest", class_names, output_dir)

    # Analyze causes of confusion and errors
    analyze_errors(y_test, y_pred_svm, "Linear SVM", class_names, output_dir)
    analyze_errors(y_test, y_pred_rf, "Random Forest", class_names, output_dir)

def analyze_errors(y_true, y_pred, model_name, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    errors = []
    
    for i, true_class in enumerate(class_names):
        for j, predicted_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                errors.append((true_class, predicted_class, cm[i, j]))

    error_analysis_path = os.path.join(output_dir, f'{model_name}_error_analysis.txt')
    with open(error_analysis_path, 'w') as f:
        f.write(f"Error Analysis for {model_name}:\n")
        f.write("True Class -> Predicted Class : Number of Instances\n")
        for error in errors:
            f.write(f"{error[0]} -> {error[1]} : {error[2]}\n")

    print(f"Error analysis saved to {error_analysis_path}")

if __name__ == "__main__":
    
    data_folders = {
        'air_conditioner': 'E:/Synopsys_Data_Challenge/data/air_conditioner',
        'car_horn': 'E:/Synopsys_Data_Challenge/data/car_horn',
        'engine_idling': 'E:/Synopsys_Data_Challenge/data/engine_idling',
        'siren': 'E:/Synopsys_Data_Challenge/data/siren'
    }
    
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
    
    class_names = list(data_folders.keys())
    X, y = load_audio_files(data_folders)
    run_job(X, y, class_names)
