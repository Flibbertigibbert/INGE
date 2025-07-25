# streamlit_app.py
import os
import streamlit as st
import time
import threading

# Import necessary LangChain components
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage # For chat history

# --- Configuration (can be put into st.secrets or passed as env vars if needed) ---
# Check if running on Streamlit Cloud or locally for API key handling
try:
    # Attempt to get API key from Streamlit secrets (for Streamlit Cloud deployment)
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (AttributeError, KeyError):
    # Fallback to environment variable (for local testing or Colab)
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Google API Key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
    st.stop() # Stop the app if API key is missing

os.environ["GOOGLE_API_KEY"] = google_api_key # Ensure it's in the environment for LangChain

VECTOR_DB_PATH = "chroma_db"
DATA_DIR = "data"
EMBEDDING_MODEL = "models/embedding-001"
GENERATIVE_MODEL = "gemini-2.5-flash-lite"

# --- Streamlit Session State Initialization ---
# This is crucial for persisting objects and chat history across reruns
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history for display and context

# --- Functions (modified for Streamlit and caching) ---

# Use st.cache_resource for expensive, global resources like LLMs and VectorDB
@st.cache_resource(show_spinner="Connecting to Gemini LLM...")
def get_gemini_llm():
    """Initializes and caches the Gemini LLM."""
    try:
        llm_instance = ChatGoogleGenerativeAI(
            model=GENERATIVE_MODEL,
            temperature=0.3,
            request_timeout=30
        )
        return llm_instance
    except Exception as e:
        st.error(f"Error initializing Gemini LLM: {e}")
        return None

@st.cache_resource(show_spinner="Loading/Creating Knowledge Base...")
def get_vector_db():
    """
    Loads or creates the ChromaDB vector store.
    This function handles both the preparation (if needed) and loading.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        request_timeout=30
    )

    # Check if the ChromaDB exists and has content
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        try:
            # Load existing DB
            db = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            st.success("Knowledge base loaded successfully!")
            return db
        except Exception as e:
            st.error(f"Error loading existing ChromaDB: {e}")
            st.warning("Attempting to re-prepare the knowledge base.")
            # Fall through to prepare knowledge base if loading fails
    else:
        st.info("ChromaDB not found or empty. Preparing knowledge base...")

    # If DB doesn't exist or loading failed, prepare it
    if not os.path.exists(DATA_DIR):
        st.warning(f"'{DATA_DIR}' directory not found. Please create it and upload PDF documents.")
        return None

    loaded_docs = load_documents_from_directory(DATA_DIR)
    if not loaded_docs:
        st.error("No documents loaded. Please ensure PDFs are in the 'data' directory.")
        return None

    document_chunks = split_documents_into_chunks(loaded_docs)
    if not document_chunks:
        st.error("No chunks created. Check document splitting configuration.")
        return None

    # Ensure the directory for ChromaDB exists before creating
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)

    db = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    st.success("Knowledge base prepared and persisted!")
    return db

def load_documents_from_directory(directory_path):
    """Loads PDF documents from the specified directory."""
    try:
        loader = PyPDFDirectoryLoader(directory_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading documents from '{directory_path}': {e}")
        return []

def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """Splits loaded documents into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

@st.cache_resource(show_spinner="Building RAG pipeline...")
def get_retrieval_chain(_llm, _vector_db):
    """Initializes and caches the RAG retrieval chain."""
    if _llm is None or _vector_db is None:
        return None

    retriever = _vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 2,
            "score_threshold": 0.1
        }
    )

    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Be concise and accurate. If the context doesn't contain enough information, "
        "say so clearly.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    Youtube_chain = create_stuff_documents_chain(_llm, prompt)
    retrieval_chain_instance = create_retrieval_chain(retriever, Youtube_chain)
    return retrieval_chain_instance

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot with Gemini Flash")
st.markdown("Ask me anything about the documents in my knowledge base!")

# Initialize chatbot components only if they haven't been already
if st.session_state.llm is None:
    st.session_state.llm = get_gemini_llm()

if st.session_state.vector_db is None:
    st.session_state.vector_db = get_vector_db()

if st.session_state.retrieval_chain is None and st.session_state.llm and st.session_state.vector_db:
    st.session_state.retrieval_chain = get_retrieval_chain(st.session_state.llm, st.session_state.vector_db)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if st.session_state.retrieval_chain:
            try:
                # Use a progress indicator for the LLM call
                with st.spinner("Thinking..."):
                    response = st.session_state.retrieval_chain.invoke({"input": prompt})
                
                answer = response.get("answer", "No answer could be generated from the available context.")
                full_response += answer

                # Display sources if available
                if "context" in response and response["context"]:
                    full_response += "\n\n**Sources:**\n"
                    for i, doc in enumerate(response["context"], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        full_response += f"- {source} (Page {page})\n"

            except Exception as e:
                full_response = f"❌ Error: {e}"
                st.error(f"Error during RAG chain invocation: {e}")
        else:
            full_response = "Chatbot not fully initialized. Please check logs for errors."
            st.warning(full_response)
        
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})