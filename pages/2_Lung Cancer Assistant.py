import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

st.set_page_config(
    page_title="Lung Cancer Assistant",
    page_icon="🤖"
)

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.load_local("faiss_lung_cancer", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})

template = """
You are an expert assistant with deep knowledge of the document.
Use the information you know from the document (without mentioning the document or context)
to answer the questions below clearly, naturally, and confidently.
If a question is not relevant, please politely respond that you do not have that information.
If the question is in a language other than English, tell the user politely to speak english.

question:
{question}

Information you know:
{context}

natural answer and direct:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5
)

# Buat custom LLMChain pakai prompt
custom_chain = load_qa_chain(
    llm=llm,  # misalnya pipeline HuggingFacePipeline
    chain_type="stuff",
    prompt=prompt
)

qa = RetrievalQA(
    combine_documents_chain=custom_chain,
    retriever=retriever,
    return_source_documents=True
)

st.title("Lung Cancer Assistant")
st.markdown('- LLM Model: gemini-2.5-flash')
st.markdown('- Embedding Model: sentence-transformers/all-MiniLM-L12-v2')
st.markdown('- Library: Langchain')
st.markdown('- Dataset: https://huggingface.co/datasets/Gaborandi/Lung_Cancer_pubmed_abstracts')

# Inisialisasi history chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user
if prompt_text := st.chat_input("Ask your question about lung cancer..."):
    # Tambahkan pesan user ke history
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Ambil context dari retriever dan generate jawaban
    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        result = qa({"query": prompt_text})
        response = result["result"]

        # Tampilkan jawaban
        st.markdown(response)

    # Simpan jawaban ke history

    st.session_state.messages.append({"role": "assistant", "content": response})


