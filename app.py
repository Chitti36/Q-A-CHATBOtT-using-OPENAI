import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()


# Set API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Load LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

def create_vector_embedding(uploaded_files):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return

    # Initialize embeddings and text splitter
    st.session_state.embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    documents = []
    for uploaded_file in uploaded_files:
        # Save to a temporary file
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())

        # Load the PDF
        loader = PyPDFLoader(f"temp_{uploaded_file.name}")
        docs = loader.load()
        documents.extend(docs)

        # Clean up temp file if you want
        os.remove(f"temp_{uploaded_file.name}")

    # Split and embed
    split_docs = text_splitter.split_documents(documents[:50])
    st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)
    st.session_state.vector_ready = True

    st.success("Vector DB created from uploaded PDFs!")


# Button to create vector DB
if st.button("Create Document Embedding"):
    create_vector_embedding(uploaded_files)

# Text input for user question
user_prompt = st.text_input("Enter your question based on uploaded documents")

# Run QA when prompt is given
if user_prompt:
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retriever_chain.invoke({'input': user_prompt})
        st.write(f"⏱️ Response time: {time.process_time() - start:.2f}s")

        st.subheader("Answer")
        st.write(response['answer'])

        with st.expander("📄 Similar Documents"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Page {i+1}**")
                st.write(doc.page_content)
                st.write("---")
    else:
        st.warning("Please upload documents and click 'Create Document Embedding' first.")

####open the link for explanation:https://chatgpt.com/share/6859995e-d160-800a-a635-2cd6930b579b