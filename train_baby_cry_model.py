# import pandas as pd
# import numpy as np
# import librosa
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import joblib

# # Function to extract audio features
# def extract_audio_features(file_name):
#     y, sr = librosa.load(file_name, duration=3.0)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfccs_mean = np.mean(mfccs.T, axis=0)
#     return mfccs_mean

# # Load the dataset from an Excel file
# # Assume 'Sheet1' contains columns named 'audio_file' and 'label'
# data = pd.read_csv('feedback.csv')  # Adjust the sheet_name as needed
# X = []  # List to hold feature vectors
# y = []  # List to hold labels

# # Extract features and labels
# for file, label in zip(data['audio_file'], data['label']):
#     try:
#         features = extract_audio_features(file)
#         X.append(features)
#         y.append(label)
#     except Exception as e:
#         print(f"Error processing {file}: {e}")

# # Convert to NumPoiy arrays
# X = np.array(X)
# y = np.array(y)

# # Train-Test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate the model
# predictions = model.predict(X_test)
# print(classification_report(y_test, predictions))

# # Save the trained model
# joblib.dump(model, 'audio_classifier.pkl')