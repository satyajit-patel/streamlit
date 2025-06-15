'''
    pip install streamlit
    streamlit run app.py
'''

import streamlit as st

st.title("Hello Brother")
player = st.selectbox('choose your fav player', ['Virat Kohli', 'MSD', 'ABD', 'Baaz', 'Rohit'])
st.write(f"you have chosen {player}. Excelent choice")
if player == 'MSD':
    st.success('you are a genius')
    st.balloons()

if st.button('click me'):
    st.balloons()

name = st.text_input('Enter your name')
if name:
    st.write(f"welcome {name}, make yourself comfortable")