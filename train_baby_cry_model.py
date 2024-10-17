import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def extract_audio_features(file_name):
    y, sr = librosa.load(file_name, duration=3.0)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def train_model():
    data = pd.read_csv('feedback.csv')
    X = np.array([extract_audio_features(file) for file in data['audio_file']])
    y = data['corrected_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save the trained model
    joblib.dump(model, 'baby_cry_classifier.pkl')

if __name__ == "__main__":
    train_model()
