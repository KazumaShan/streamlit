import streamlit as st
import librosa

#input voice

#------------------

#app title
st.title('Baby Crying Voice Detection')


#write text
#display text output
st.write('To find out why baby is crying???')


def extract_audio_features(uploaded_file):
   y, sr = librosa.load(uploaded_file)






