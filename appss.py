import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from htmlTemplates import css, bot_template, user_template
import base64
import io

# Load environment variables (locally) or Streamlit Secrets (in cloud)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🚨 OpenAI API key not found. Please set it in .env or Streamlit secrets.")
    st.stop()


def get_pdf_text_and_chunks(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": i + 1}))
    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    answer = response['answer']
    source_docs = response.get('source_documents', [])
    pages = sorted(set(doc.metadata.get("page", "?") for doc in source_docs))
    page_info = f"<div style='color: gray; font-size: small;'>Pages referenced: {', '.join(map(str, pages))}</div>"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "user": user_question,
        "bot": answer,
        "pages": pages
    })

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer + page_info), unsafe_allow_html=True)


def summarize_documents(vectorstore):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
    documents = list(vectorstore.docstore._dict.values())
    return summarize_chain.run(documents)


def display_pdf(file):
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    st.markdown(f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
                width="100%" height="700px" type="application/pdf"></iframe>
    """, unsafe_allow_html=True)


def download_chat_history():
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.warning("No chat history to download.")
        return

    output = io.StringIO()
    for i, chat in enumerate(st.session_state.chat_history, 1):
        output.write(f"Q{i}: {chat['user']}\n")
        output.write(f"A{i}: {chat['bot']}\n")
        output.write(f"Pages referenced: {', '.join(map(str, chat['pages']))}\n")
        output.write("-" * 40 + "\n")

    st.download_button(
        label="📥 Download Chat History",
        data=output.getvalue(),
        file_name="chat_history.txt",
        mime="text/plain"
    )


def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("<h1 style='text-align: center;'>📚 Chat with your PDFs</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        with st.container(border=True, height=700):
            st.subheader("📂 Upload & Interact")
            pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
            if st.button("Process") and pdf_docs:
                with st.spinner("Processing PDFs..."):
                    docs = get_pdf_text_and_chunks(pdf_docs)
                    chunks = split_documents(docs)
                    vectorstore = get_vectorstore(chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Ready to chat!")
                    st.session_state.chat_history = []

            question = st.text_input("Ask a question:")
            if question and st.session_state.conversation:
                handle_userinput(question)

            if st.button("Summarize") and st.session_state.vectorstore:
                with st.spinner("Generating summary..."):
                    summary = summarize_documents(st.session_state.vectorstore)
                st.markdown(f"<div class='justified'><strong>Summary:</strong><br>{summary}</div>", unsafe_allow_html=True)

            st.markdown("---")
            download_chat_history()

    with col2:
        with st.container(border=True, height=700):
            st.subheader("📄 PDF Preview")
            if 'vectorstore' in st.session_state and pdf_docs:
                for pdf in pdf_docs:
                    with st.expander(pdf.name, expanded=True):
                        display_pdf(pdf)


if __name__ == '__main__':
    main()
