import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
#input voice
import librosa
import uuid
import os
import joblib
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#------------------
# Load the model at the start of the app
model = None
if os.path.exists('baby_cry_classifier.pkl'):
    model = joblib.load('baby_cry_classifier.pkl')
else:
    st.warning("Model not found. Please train the model first.")

st.title('Baby Crying Voice Detection')

# Add retrain button
if st.button('Retrain Model'):
    # Start the training script as a subprocess
    result = subprocess.run(['python', 'train_baby_cry_model.py'], capture_output=True, text=True)
    st.text(result.stdout)
    if result.returncode == 0:
        st.success("Model retrained successfully.")
    else:
        st.error("Error during model retraining. Check logs for details.")

#------------------

#app title
st.title('Baby Crying Voice Detection')


#write text
#display text output
st.write('To find out why baby is crying???')

#-------------extract audio-----------------------------------------------------------

def extract_audio_features(uploaded_file):
  y, sr = librosa.load(uploaded_file)

  # Extract basic features
  # Compute the zero-crossing rate (an example feature, indicating pitch and voice changes)
  zcr = librosa.feature.zero_crossing_rate(y)

  # Calculate the root-mean-square energy (indicates loudness)
  rmse = librosa.feature.rms(y=y)

  # Compute spectral centroid (indicates brightness,pitch)
  spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

  #Calculate duration
  duration = librosa.get_duration(y=y, sr=sr)
  
  return {
      "zcr": zcr.mean(),
      "rmse": rmse.mean(),
      "spectral_centroids": spectral_centroids.mean(),
      "duration": duration
  }

#------------Apply the Heuristic Rules------------------------------------------------
def rule_based_classification(features):
    zcr = features["zcr"]
    rmse = features["rmse"]
    spectral_centroids = features["spectral_centroids"]
    duration = features["duration"]

    # Example logic using feature conditions
    if spectral_centroids > 1700 and duration >= 1.0 and zcr < 0.1:
        return "Hungry"
    elif spectral_centroids > 1600 and duration >= 0.5 and zcr < 0.05:
        return "Burp"
    elif spectral_centroids < 1500 and duration < 0.5:
        return "Discomfort"
    elif spectral_centroids < 1400 and duration > 2.0:
        return "Lower gas"
    elif spectral_centroids > 1300 and duration > 1.0:
        return "Sleepy/Tired"
    else:
        return "Unknown"

#------------Store feedback in CSV------------------------------------------------
def store_feedback(audio_file_name, initial_prediction, corrected_label, feedback_file='feedback.csv'):
    if os.path.exists(feedback_file):
        feedback_data = pd.read_csv(feedback_file)
    else:
        feedback_data = pd.DataFrame(columns=['audio_file', 'initial_prediction', 'corrected_label'])

    # Create a DataFrame for the new data to append
    new_data = pd.DataFrame([{
        'audio_file': audio_file_name,
        'initial_prediction': initial_prediction,
        'corrected_label': corrected_label
    }])

    # Use pd.concat to concatenate the new_data DataFrame with the existing feedback_data DataFrame
    feedback_data = pd.concat([feedback_data, new_data], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    feedback_data.to_csv(feedback_file, index=False)

# Initialize session state
if 'corrected_label' not in st.session_state:
    st.session_state['corrected_label'] = None


#---------------------------------------------

#upload voice
# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Create a unique filename
    audio_file_name = f"{uuid.uuid4()}.wav"
    with open(audio_file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display audio player
    st.audio(uploaded_file)

    # Extract features and get initial prediction
    features = extract_audio_features(audio_file_name)
    initial_prediction = rule_based_classification(features)

    # Display the initial prediction
    st.write(f"The predicted reason for crying is: {initial_prediction}")
    correct = st.radio("Is this correct?", options=["Please select Yes/No", "Yes", "No"])

    if correct == "No":
        # Provide selection options with a placeholder
        label_options = ["Select a label", "Neh (Hungry)", "Owh (Sleepy/Tired)", "Heh (Discomfort)", "Eair (Lower gas)", "Eh (Burp)"]
        corrected_label = st.selectbox("Select the correct label if known:", label_options)

        if st.button('Submit Correction') and corrected_label != "Select a label":
            st.write(f"Stored corrected label: {corrected_label}")
            store_feedback(audio_file_name, initial_prediction, corrected_label)
            st.success("Feedback saved.")
        elif corrected_label == "Select a label":
            st.warning("Please select a valid label before submitting.")

    elif correct == "Yes":
        store_feedback(audio_file_name, initial_prediction, initial_prediction)
        st.success("Prediction confirmed and saved.")

# Add download button for feedback.csv
if os.path.exists('feedback.csv'):
    with open('feedback.csv', 'rb') as f:
        st.download_button(
            label="Download Feedback CSV",
            data=f,
            file_name='feedback.csv',
            mime='text/csv'
        )

#-------------------------------------------




