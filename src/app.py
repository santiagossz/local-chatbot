import uuid

import streamlit as st
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from assistant import get_rag_assistant, get_rag_history_assistant, get_vector_store, get_message_history, get_session_ids

st.set_page_config(page_title="Santiago RAG")
st.title("Santiago simple RAG")

def main() -> None:
    # Chat session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_config = {"configurable": {"session_id": st.session_state.session_id}}
    
    # Get all sessions 
    st.session_state.sessions = get_session_ids()
    if st.session_state.session_id not in st.session_state.sessions:
        st.session_state.sessions.insert(0,st.session_state.session_id)
     
    # Get Embedding model
    embeddings_models = ["embed-english-v3.0","nomic-embed-text"]
    if "embeddings_model" not in st.session_state:
        st.session_state.embeddings_model = embeddings_models[1]
    st.sidebar.selectbox("Select Embedding:",
                        options = embeddings_models,
                        on_change = clear_docs_ui,
                        key="embeddings_model"
                        )        
    
    # Get RAG chain w/ memory
    collection_name = f"santiago_rag_{st.session_state.session_id}_{st.session_state.embeddings_model}"
    vector_store = get_vector_store(st.session_state.embeddings_model,collection_name)
    rag_assistant  = get_rag_assistant(vector_store)
    history_rag_assitant = get_rag_history_assistant(rag_assistant)
    
    # History store in streamlit session
    history_store = StreamlitChatMessageHistory(key="messages")
    history_store.messages =  get_message_history(st.session_state.session_id).messages
    if len(history_store.messages) == 0:
        history_store.add_ai_message("Upload a doc and ask me questions...")
    
    # Show all historical messages
    for message in history_store.messages:
        st.chat_message(message.type).write(message.content)
    # Ask question and return answer
    if question:= st.chat_input():
        st.chat_message("human").write(question)
        response = history_rag_assitant.invoke({"input":question},config=session_config)
        st.chat_message("ai").write(response["answer"])
        
    ## doc names & VS ids dicts
    ## docs per embedding model per session
    if f"added_docs" not in st.session_state:
        st.session_state.added_docs = {s:{e:[]  for e in embeddings_models} for s in st.session_state.sessions}
    if "documents_ids" not in st.session_state:
        st.session_state.documents_ids = {s:{e:{}  for e in embeddings_models} for s in st.session_state.sessions}
    
    # Add URL document
    if "url_uploader_key" not in st.session_state:
        st.session_state.url_uploader_key = -1
    input_url = st.sidebar.text_input("Add data from URL", key=st.session_state.url_uploader_key)
    if st.sidebar.button("Add URL"):
        if input_url:
            processing_info = st.sidebar.info("Processing URL...", icon ="ℹ️") 
            try:
                loader = UnstructuredURLLoader(urls=[input_url],
                                               headers={"User-Agent":"My User Agent 1.0"})
                add_docs(loader,vector_store,input_url)
                st.sidebar.success("Successfully loaded documents", icon="✅")
            except:
                st.sidebar.error("Unable to load documents", icon="❗")
            processing_info.empty()
            
    # Add PDF
    added_pdfs = st.session_state.added_docs[st.session_state.session_id][st.session_state.embeddings_model]
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    uploaded_pdf = st.sidebar.file_uploader(
        "Add a PDF :page_facing_up:", type="pdf", key=st.session_state.file_uploader_key)
    if uploaded_pdf:
        alert = st.sidebar.info("Processing PDF...", icon="🧠")
        pdf_name = uploaded_pdf.name
        if pdf_name not in added_pdfs:
            with open(pdf_name, mode="wb") as w:
                w.write(uploaded_pdf.getvalue())
            loader = PyPDFLoader(pdf_name)
            add_docs(loader,vector_store,pdf_name)            
        alert.empty()
    
    # Show added docs with button to delete from ui and VS 
    st.sidebar.header("Added documents")
    for doc in added_pdfs:
        doc_container = st.sidebar.container()
        if doc_container:
            with doc_container:
                cols = st.columns([4,1])
                cols[0].write(doc)
                cols[1].button("❌",key=f"{doc}_{st.session_state.session_id}_{st.session_state.embeddings_model}",
                               on_click=clear_uploaded_docs, args=[doc,vector_store])


    st.sidebar.button("Clear Updated Documents",on_click=delete_docs,
                         args=[vector_store])
    
    st.sidebar.selectbox("Chat ID",options=st.session_state.sessions,
                                      index=st.session_state.sessions.index(st.session_state.session_id), 
                                      on_change=clear_docs_ui,
                                      key= "session_id")
    st.sidebar.button("New Chat",on_click=new_session_id, args=[embeddings_models])
        
   

def add_docs(loader, vector_store,doc):
    text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=16)
    docs = loader.load_and_split(text_splitter)
    documents_ids = vector_store.add_documents(docs)
    st.session_state.documents_ids[st.session_state.session_id][st.session_state.embeddings_model][doc] = documents_ids
    st.session_state.added_docs[st.session_state.session_id][st.session_state.embeddings_model].append(doc)

def clear_docs_ui():
    st.session_state.file_uploader_key +=1
    st.session_state.url_uploader_key -= 1

def delete_docs(vector_store):
    clear_docs_ui() 
    st.session_state.added_docs[st.session_state.session_id][st.session_state.embeddings_model] = []
    vector_store.delete_collection()  
    
def clear_uploaded_docs(doc, vector_store):
    clear_docs_ui()
    st.session_state.added_docs[st.session_state.session_id][st.session_state.embeddings_model].remove(doc)
    vector_store.delete(
        st.session_state.documents_ids[st.session_state.session_id][st.session_state.embeddings_model][doc]
        )
    
def new_session_id(embeddings_models):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.added_docs[st.session_state.session_id]={e:[]  for e in embeddings_models}
    st.session_state.documents_ids[st.session_state.session_id]={e:{}  for e in embeddings_models}
  

    
main()
