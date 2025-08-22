import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from langchain_core.documents import Document

# ------------------------------
# 1. Configure Streamlit Page
# ------------------------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="📑")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ------------------------------
# 2. Set up sidebar and main interface
# ------------------------------
st.title("📑 HR Policy Assistant")
st.write("Upload your HR policy document and ask questions about it!")

# Check if Ollama is available
def check_ollama():
    try:
        # Try to initialize Ollama
        embeddings = OllamaEmbeddings(model="llama2")
        embeddings.embed_query("test")
        return True
    except Exception as e:
        st.error(f"Ollama error details: {str(e)}")
        return False

if not check_ollama():
    st.error("⚠️ Cannot connect to Ollama!")
    st.info("""
    Please make sure:
    1. Ollama is installed (visit https://ollama.ai/download)
    2. Ollama service is running
    3. Llama2 model is pulled
    
    Run these commands in terminal:
    ```
    # Start Ollama service
    ollama serve
    
    # In another terminal, pull Llama2
    ollama pull llama2
    ```
    """)
    st.stop()

with st.sidebar:
    st.title("Document Upload")
    uploaded_file = st.file_uploader("Upload your HR Policy PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load and split the document
            pdf_reader = PdfReader("temp.pdf")
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_text(text)
            splits = [Document(page_content=t) for t in texts]
            
            try:
                # Create embeddings and store them
                with st.spinner("Creating embeddings with Llama2..."):
                    embeddings = OllamaEmbeddings(model="llama2")
                    st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success("Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.info("Please make sure Ollama is running and Llama2 model is available")
                st.session_state.vector_store = None

# ------------------------------
# 3. Chat Interface
# ------------------------------
if st.session_state.vector_store is None:
    st.info("Please upload an HR policy document to start chatting!")
else:
    try:
        # Set up the QA chain
        llm = ChatOllama(
            model="llama2",
            temperature=0
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(),
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error setting up chat: {str(e)}")
        st.info("""
        Please make sure:
        1. Ollama service is running (`ollama serve`)
        2. Llama2 model is available (`ollama pull llama2`)
        """)
        st.stop()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your HR policies"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain({"query": prompt})
                    response = result["result"]
                    
                    # Display the main response
                    st.markdown(response)
                    
                    # Display sources if available
                    if "source_documents" in result:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**Source {i+1}:**\n{doc.page_content}\n")
                    
                    # Update chat history
                    st.session_state.chat_history.extend([(prompt, response)])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.markdown("I apologize, but I encountered an error while processing your question. Please try again.")
