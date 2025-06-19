import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']
if not GROQ_API_KEY:
    st.error("API key missing")
    st.stop()

st.title("Chat Bot")

# Setup model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY
)

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(SystemMessage("You are a helpful assistant which gives concise answers."))

# Prompt input
prompt = st.text_input("Enter prompt", key="unique_prompt")

if prompt:
    st.session_state.chat_history.append(HumanMessage(prompt))
    response = llm.invoke(st.session_state.chat_history)
    ai_msg = response.content
    st.session_state.chat_history.append(AIMessage(ai_msg))
    st.write("**AI:**", ai_msg)
