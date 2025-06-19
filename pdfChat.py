import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

if not (GROQ_API_KEY or GOOGLE_API_KEY):
    st.error("API key missing")
    st.stop()

st.title("PDF Chat")

# Setup model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GROQ_API_KEY
)

file = st.file_uploader('Upload PDF', type='pdf')

if file:
    # step 1 (upload PDF)
    if file.type != 'application/pdf':
        st.write('Please upload a valid PDF file.')
    else:
        # Save uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        # Load and process PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # st.write(f"Total pages: {len(pages)}")
        # st.write(f"type: {type(pages)}")
        # st.write(pages)
        # st.write(pages[0].page_content)

        # Extract raw text from each page
        raw_texts = [page.page_content for page in pages]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        # Now pass this list of strings to the splitter
        documents = text_splitter.create_documents(raw_texts)
        # st.write(f"type: {type(documents)}") # list of Documents
        # st.write(documents)


        # step 2 (make embeddings)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")




        # step 3 (store embeddings)
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
        vector_store.add_documents(documents=documents)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2}) # #k i.e top 2 relevent chunk


        # take user query => take similar content => pass it with query to LLM
        # https://satyajit-gen-ai.hashnode.dev/basic-rag-for-pdf-chat-short-and-crisp
        query = st.text_input("Enter prompt")
        if query:
            context = retriever.invoke(query)

            prompt_template = ChatPromptTemplate([
                ("system", "you are a helpful AI asistant who responds only based on this context {context}"),
                ("user", "give a short and crisp answer of this query {query}")
            ])
            prompt = prompt_template.invoke({"context": context, "query": query})
            st.write(prompt)
            response = llm.invoke(prompt)
            aiMessage = response.content
            st.write(aiMessage)
