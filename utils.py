from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

def preprocess_text(text):

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = text.lower()
    return text

def load_pdf_to_chroma(data_directory="data", persist_directory="data/vectorstore"):
    
    loader = DirectoryLoader(data_directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

  
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=persist_directory  
    )
    vectorstore.persist()  
    return vectorstore

# Exemple d'utilisation
if __name__ == "__main__":
    vectorstore = load_pdf_to_chroma()
    print("Base vectorielle créée et sauvegardée dans 'data/vectorstore'.")
