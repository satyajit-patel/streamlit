import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# from src.helpers.displayInstructions import showInstructions
# from src.helpers.checkKeyExist import isKeyExist

# --- API Key Instructions ---
api_guide = """
### How to get your Groq API Key:
1. Visit [Groq Console](https://console.groq.com/keys).
2. Sign up or log in with your account.
3. Navigate to the **API Keys** section.
4. Click on **+ Create Key** to generate a new API key.
5. Copy the key and paste it in your `.env` or secret config.
"""

@st.cache_resource(show_spinner=True)
def load_model():
    # api_check = isKeyExist("GROQ_API_KEY", "api_key")
    # if not api_check["GROQ_API_KEY"]:
    #     showInstructions(markdown_text=api_guide, fields="GROQ_API_KEY")
    #     st.stop()

    api_key = os.getenv("GROQ_API_KEY") or st.secrets["api_key"]["GROQ_API_KEY"]
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )

def chatWithYourPDF():
    st.title("📄 Chat With Your PDF")

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if file and file.type == "application/pdf":
        with st.spinner("Processing your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            raw_texts = [page.page_content for page in pages]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )

            documents = text_splitter.create_documents(raw_texts)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
            vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
            st.session_state.retriever = vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 2}
            )

            st.success("✅ PDF processed successfully. You can now start chatting.")

    elif file and file.type != "application/pdf":
        st.error("Please upload a valid PDF file.")

    if st.session_state.retriever:
        query = st.text_input("Ask something about the PDF...")
        if query:
            with st.spinner("Thinking..."):
                context = st.session_state.retriever.invoke(query)
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Use the following context to answer the user's query: {context}"),
                    ("user", "{query}")
                ])
                prompt = prompt_template.invoke({"context": context, "query": query})
                llm = load_model()
                response = llm.invoke(prompt)
                ai_message = response.content
                st.markdown("#### 📌 Response")
                st.write(ai_message)
