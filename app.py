import streamlit as st
from main import ask_question, generate_general_questions, evaluate_responses

st.title("Chatbot basé sur un document PDF")

st.header("Niveau 1 : Posez une question sur le document")

user_query = st.text_input("Posez votre question sur le document")

# niveau 1:
if user_query:
    response = ask_question(user_query)
    st.subheader("Réponse du chatbot :")
    st.write(response)

# Niveau 2 : 
st.header("Niveau 2 : Génération de questions générales")

if st.button("Générer des questions générales"):
    try:
        questions = generate_general_questions()
        if questions:
            st.session_state.questions = questions  
            st.subheader("Questions générées :")
            for i, q in enumerate(questions, start=1):
                st.write(f"{i}. {q}")
        else:
            st.write("Aucune question générée.")
    except Exception as e:
        st.write(f"Erreur lors de la génération des questions : {e}")

# Niveau 3 : 
st.header("Niveau 3 : Évaluation des réponses et plan de formation")

if 'questions' in st.session_state and st.session_state.questions:
    user_responses = {}
    for q in st.session_state.questions:
        response = st.text_input(f"Votre réponse à la question : {q}")
        if response:
            user_responses[q] = response
    
    if st.button("Évaluer les réponses"):
        if user_responses:
            try:
                evaluation_plan = evaluate_responses(user_responses)
                st.subheader("Plan de formation suggéré :")
                st.write(evaluation_plan)
            except Exception as e:
                st.write(f"Erreur lors de l'évaluation des réponses : {e}")
        else:
            st.write("Aucune réponse à évaluer.")
else:
    st.write("Aucune question générée pour l'évaluation.")
