import streamlit as st
import time

#st.ballons()
#st.subheader("Progress bar")
#st.progress(5)

#st.checkbox('yes')
#st.button('click')
st.title('LOGIN')
st.text_input('Guardian Name')
st.radio('Relationship with Baby',['MOM', 'DAD','UNCLE','AUNTY', 'OTHERS'])
st.text_input('Email address')

st.divider()
st.text_input('Baby Name')
st.radio('Pick The Babies Gender',['Male', 'Female'])
st.slider('Baby age based on Month', 0, 48)
