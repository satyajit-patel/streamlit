import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-left: 2rem;
        text-align: right;
    }
    .ai-message {
        background-color: #e9ecef;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-right: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Validate API key
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    st.error("🔑 GROQ API key is missing. Please check your .env file.")
    st.info("💡 Make sure your .env file contains: GROQ_API_KEY=your_api_key_here")
    st.stop()

# App header
st.title("🤖 AI Chat Assistant")
st.markdown("*Powered by Llama 3.1 via Groq*")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(
        SystemMessage("You are a helpful assistant that provides concise, accurate, and friendly responses.")
    )

if "message_count" not in st.session_state:
    st.session_state.message_count = 0

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Setup model with error handling
@st.cache_resource
def initialize_model():
    try:
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1000,
            timeout=30,
            max_retries=3,
            api_key=GROQ_API_KEY
        )
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        return None

llm = initialize_model()

if not llm:
    st.stop()

# Function to display chat history
def display_chat_history():
    if len(st.session_state.chat_history) > 1:  # Skip system message
        st.markdown("### 💬 Chat History")
        chat_html = '<div class="chat-container">'
        
        for message in st.session_state.chat_history[1:]:  # Skip system message
            if isinstance(message, HumanMessage):
                chat_html += f'<div class="user-message">👤 You: {message.content}</div>'
            elif isinstance(message, AIMessage):
                chat_html += f'<div class="ai-message">🤖 AI: {message.content}</div>'
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

# Function to handle user input
def process_user_input(user_input):
    if not user_input or not user_input.strip():
        return
    
    user_input = user_input.strip()
    
    # Check for inappropriate content or potential issues
    if len(user_input) > 1000:
        st.warning("⚠️ Message too long. Please keep it under 1000 characters.")
        return
    
    try:
        st.session_state.is_processing = True
        
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(user_input))
        
        # Show processing indicator
        with st.spinner("🤔 Thinking..."):
            # Get AI response
            response = llm.invoke(st.session_state.chat_history)
            ai_response = response.content
        
        # Add AI response to history
        st.session_state.chat_history.append(AIMessage(ai_response))
        st.session_state.message_count += 1
        
        # Show success message briefly
        success_placeholder = st.empty()
        success_placeholder.success("✅ Response generated successfully!")
        time.sleep(1)
        success_placeholder.empty()
        
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        st.info("💡 Please try again or check your internet connection.")
        
        # Remove the user message if AI response failed
        if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
            st.session_state.chat_history.pop()
    
    finally:
        st.session_state.is_processing = False

# Sidebar with controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Chat statistics
    st.metric("Messages Exchanged", st.session_state.message_count)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = [
            SystemMessage("You are a helpful assistant that provides concise, accurate, and friendly responses.")
        ]
        st.session_state.message_count = 0
        st.success("Chat history cleared!")
        st.rerun()
    
    # Export chat button
    if len(st.session_state.chat_history) > 1:
        chat_export = ""
        for i, message in enumerate(st.session_state.chat_history[1:], 1):
            if isinstance(message, HumanMessage):
                chat_export += f"User {i//2 + 1}: {message.content}\n\n"
            elif isinstance(message, AIMessage):
                chat_export += f"AI {i//2 + 1}: {message.content}\n\n"
        
        st.download_button(
            label="📥 Download Chat",
            data=chat_export,
            file_name=f"chat_history_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("• Ask clear, specific questions")
    st.markdown("• Keep messages under 1000 characters")
    st.markdown("• Use the clear button to start fresh")

# Main chat interface
col1, col2 = st.columns([4, 1])

with col1:
    # Use a form to handle enter key properly
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "💭 Type your message here...",
            placeholder="Ask me anything!",
            disabled=st.session_state.is_processing,
            key="user_input"
        )
        submit_button = st.form_submit_button(
            "Send 📤", 
            disabled=st.session_state.is_processing,
            type="primary"
        )

with col2:
    # Quick action buttons
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    if st.button("💡 Suggest Topics", disabled=st.session_state.is_processing):
        suggestions = [
            "What's the weather like?",
            "Explain quantum computing",
            "Write a Python function",
            "Tell me a joke",
            "Help me plan my day"
        ]
        st.info("💡 **Try asking:**\n" + "\n".join([f"• {s}" for s in suggestions]))

# Process input when form is submitted
if submit_button and user_input:
    process_user_input(user_input)
    st.rerun()

# Display chat history
display_chat_history()

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit and LangChain | "
    f"Total messages: {st.session_state.message_count}"
)