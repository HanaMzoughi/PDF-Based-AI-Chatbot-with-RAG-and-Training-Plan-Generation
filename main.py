from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import os
import traceback


huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingface_token:
    raise ValueError("Le jeton Hugging Face n'est pas défini.")

pdf_path = "data/document.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load_and_split()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
chroma_directory = "data/vectorstore"
chroma_db = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)

if len(chroma_db.get()["ids"]) == 0:
    print("Initialisation de la base de données vectorielle avec les documents...")
    chroma_db = Chroma.from_documents(docs, embeddings, persist_directory=chroma_directory)
    chroma_db.persist()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=huggingface_token,
    temperature=0.3
)
retriever = chroma_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Niveau 1 : 
def ask_question(query):
    
    try:
        response = qa_chain.run(query)
        return response.strip() if response else "Aucune réponse valide obtenue."
    except Exception as e:
        print(f"Erreur dans 'ask_question': {e}")
        traceback.print_exc()
        return "Erreur lors de la consultation du document."

# Niveau 2 : 
def generate_general_questions():
    
    try:
        context = qa_chain.run("Quels sont les principaux sujets abordés dans le document ?")
        if not context:
            raise ValueError("Impossible de récupérer un contexte valide pour générer des questions.")
        
        prompt = (
            f"Voici les principaux sujets abordés dans un document : {context}. "
            f"Générez cinq questions ouvertes pour tester la compréhension du document."
        )
        response = llm.generate([prompt])
        questions = response.generations[0][0].text.split("\n") if response.generations else []
        return [q.strip() for q in questions[:5] if q.strip()]
    except Exception as e:
        print(f"Erreur dans 'generate_general_questions': {e}")
        traceback.print_exc()
        return []

# Niveau 3 : 
def evaluate_responses(user_responses):
    
    try:
        if not user_responses:
            raise ValueError("Aucune réponse utilisateur fournie pour évaluation.")

        formatted_responses = "\n".join([f"- {q}: {resp}" for q, resp in user_responses.items()])
        context = qa_chain.run("Quels sont les points importants pour valider les réponses ?")
        if not context:
            raise ValueError("Impossible de récupérer un contexte valide pour évaluer les réponses.")
        
        prompt = (
            f"En vous basant sur : {context}, évaluez : {formatted_responses}. "
            "Proposez un plan de formation structuré pour combler les lacunes."
        )
        response = llm.generate([prompt])
        return response.generations[0][0].text.strip() if response.generations else "Aucune évaluation générée."
    except Exception as e:
        print(f"Erreur dans 'evaluate_responses': {e}")
        traceback.print_exc()
        return "Erreur lors de l'évaluation des réponses utilisateur."

# teste des trois niveaux
if __name__ == "__main__":
    # Niveau 1 : 
    print("=== Niveau 1 : Consultation du document ===")
    user_query = input("Posez votre question sur le document : ")
    level1_response = ask_question(user_query)
    print(f"Réponse : {level1_response}")

    # Niveau 2 : 
    print("\n=== Niveau 2 : Génération de questions générales ===")
    questions = generate_general_questions()
    if questions:
        print("Questions générées :")
        for i, q in enumerate(questions, start=1):
            print(f"{i}. {q}")
    else:
        print("Aucune question n'a été générée.")

    # Niveau 3 : 
    if questions:
        print("\n=== Niveau 3 : Évaluation des réponses ===")
        user_responses = {}
        for q in questions:
            user_responses[q] = input(f"{q}\nVotre réponse : ")
        
        if user_responses:
            evaluation = evaluate_responses(user_responses)
            print("\nPlan de formation suggéré :")
            print(evaluation)
        else:
            print("Aucune réponse utilisateur à évaluer.")
    else:
        print("Impossible de passer au Niveau 3 sans questions générées au Niveau 2.")
