import streamlit as st

#set the app title
st.title('My first stremlit app')

#streamlit run [filename].py
#streamlit run main.py

#write text
#display text output
st.write('Welcome to my first streamlit app')

#display a button
st.button("Reset", type="primary")
if st.button("Say Hello"):
  st.write("Why hello there")
else:
  st.write("Goodbye")