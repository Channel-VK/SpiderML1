import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="RAG Search Engine", page_icon="📚", layout="centered")
st.title("RAGgle")
error_container = st.empty()
st.caption("Stop digging through documents. Start searching in human language.")

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    loader = PyPDFDirectoryLoader("papers/")
    documents = loader.load()

    if not documents:
        # This pushes the error to the top billboard, safely away from spinners
        error_container.error("Error: Found 0 pages of text! Either 'papers/' is empty, the terminal is in the wrong folder(cd to the correct folder lol).")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    return db


@st.cache_resource
def load_llm():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=1,  # Low temp so it doesn't hallucinate facts
        max_tokens=1024,
        api_key=api_key
    )


vector_db = load_vector_db()
llm = load_llm()
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "AI", "content": "Hello! I have loaded your NLP papers. What would you like to know?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask any question about the papers..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        with st.spinner("Searching papers..."):
            docs = retriever.invoke(user_input)
            context_text = "\n\n".join([doc.page_content for doc in docs])

            template = """
            You are an expert AI research assistant. Use the following pieces of retrieved context to answer the user's question. 
            If you don't know the answer based strictly on the context, just say 'I cannot find the answer in the provided NLP papers.' 
            Do not hallucinate or use outside knowledge.

            Context: {context}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)

            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context_text, "question": user_input})

            citations = set([
                                f"Page {doc.metadata.get('page', 'Unknown')} of {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
                                for doc in docs])
            citation_text = "\n\n**Sources Used:**\n" + "\n".join([f"- {c}" for c in citations])

            final_response = answer + citation_text

            st.markdown(final_response)

    st.session_state.messages.append({"role": "AI", "content": final_response})