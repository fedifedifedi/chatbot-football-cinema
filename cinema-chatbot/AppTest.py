import streamlit as st
from transformers import pipeline

# Titre de l'application
st.title("🎬 Analyse de Sentiment avec DistilBERT")
st.write("Entrez une critique de film et découvrez si elle est positive ou négative.")

# Charger le modèle une seule fois avec mise en cache
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="./distilbert_imdb_model")

# Charger le modèle
try:
    classifier = load_model()
except Exception as e:
    st.error("Erreur lors du chargement du modèle. Assurez-vous qu'il est bien sauvegardé.")
    st.stop()

# Zone de saisie utilisateur
user_input = st.text_area("Entrez votre critique ici :", height=200)

# Bouton pour déclencher l'analyse
if st.button("🔍 Analyser"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer une critique avant de valider.")
    else:
        with st.spinner("Analyse en cours..."):
            result = classifier(user_input)
            label = "👍 Positif" if result[0]['label'] == 'LABEL_1' else "👎 Négatif"
            score = round(result[0]['score'] * 100, 2)
            st.success(f"Résultat : {label} ({score}% de confiance)")
