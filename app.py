from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_groq import ChatGroq
from operator import itemgetter

import streamlit as st
import tempfile
import os
import pandas as pd
import getpass
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv() 
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in environment variables or .env file")
    st.stop()

# Ensure the API key is set
# if not os.environ.get("GROQ_API_KEY"):
#     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

st.set_page_config(page_title="QA")
st.title("QA Assistant")

@st.cache_resource(ttl="1h")
def retriever1(uploaded_file):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_file:
        temp_fpath = os.path.join(temp_dir.name, file.name)
        with open(temp_fpath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_fpath)
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)
    embeddings_model = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(doc_chunks, embeddings_model)
    retriever = vector.as_retriever()
    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF",
    type=["pdf"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF docs to continue.")
    st.stop()

retriever = retriever1(uploaded_files)
llm = ChatGroq(model_name="llama3-8b-8192")

qa_temp = """
Use only the following pieces of context and chat history to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer as concise as possible.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_temp)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


qa_rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question")
    }
    | qa_prompt
    | llm
)

streamlit_msg_his = StreamlitChatMessageHistory(key="langchain_messages")
if len(streamlit_msg_his.messages) == 0:
    streamlit_msg_his.add_ai_message("Ask a question related to uploaded documents")


for msg in streamlit_msg_his.messages:
    st.chat_message(msg.type).write(msg.content)

class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write):
        super().__init__()
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents:
            metadata = {
                "source": d.metadata["source"],
                "page": d.metadata["page"],
                "content": d.page_content[:200]
            }
            idx = (metadata["source"], metadata["page"])
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, documents, *, run_id, parent_run_id, **kwargs):
        if self.sources:
            st.markdown("__Sources__" + "\n")
            st.dataframe(data=pd.DataFrame(self.sources), width=1000)

if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)
    streamlit_msg_his.add_user_message(user_prompt)
    
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        sources_container = st.write("")
        pm_handler = PostMessageHandler(sources_container)
        config = {'callbacks': [stream_handler, pm_handler]}
        
        chat_history_text = "\n".join(
            [f"Human: {msg.content}" if msg.type == "human" else f"AI: {msg.content}" 
             for msg in streamlit_msg_his.messages]
        )
        
        response = qa_rag_chain.invoke(
            {"question": user_prompt, "chat_history": chat_history_text},
            config
        )
        st.write(response.content)
        streamlit_msg_his.add_ai_message(response.content)
