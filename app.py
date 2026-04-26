import streamlit as st
import os
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# 🔑 API key
os.environ["OPENAI_API_KEY"] =""


# 🤖 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 AI PDF Chatbot (RAG)")

st.sidebar.title("📂 Upload & Settings")

# 🧠 Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🚀 Cached processing
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFium2Loader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)

    return db

# 📤 Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file_path = f"temp_{file_hash}.pdf"

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    st.success("PDF uploaded successfully!")

    db = process_pdf(file_path)

    st.success("Document processed successfully!")

    # 📄 Summary button
    if st.button("📄 Summarize Document"):
        results = db.similarity_search("summarize document", k=10)
        context = "\n".join([doc.page_content for doc in results])

        prompt = f"""
        Provide a clear summary of the document.

        Context:
        {context}
        """

        response = llm.invoke(prompt)

        st.subheader("📄 Summary")
        st.write(response.content)

    st.markdown("## 💬 Chat with your document")

    # 🧾 Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # 💬 Chat input (NEW)
    user_query = st.chat_input("Ask something...")

    if user_query:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        results = db.similarity_search_with_score(user_query, k=8)

        filtered_docs = []
        for doc, score in results:
            if score < 0.5:
                filtered_docs.append(doc)

        if not filtered_docs:
            answer = "I don't know."
        else:
            context = "\n".join([doc.page_content for doc in filtered_docs])

            prompt = f"""
            You are a helpful assistant.

            If the question is about overall document (like summary or topic),
            then analyze all context and give a high-level summary.

            Otherwise:
            - Answer ONLY from context
            - If not found → say "I don't know"

            Context:
            {context}

            Question: {user_query}
            """

            response = llm.invoke(prompt)
            answer = response.content

        # Store assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(answer)

        # 📚 Source chunks
        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(filtered_docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:300])
                st.write("---")