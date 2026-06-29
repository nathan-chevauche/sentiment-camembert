import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Analyse de sentiment")

st.title("Analyse de sentiment : Avis de films français")
st.write(
    "Modèle CamemBERT fine-tuné sur le dataset Allociné "
    "(97.58% accuracy, F1 0.9758 sur le test set)."
)

text = st.text_area(
    "Entrez un avis de film en français :",
    placeholder="Ex : Ce film est vraiment magnifique, je recommande !",
    height=100,
)

if st.button("Analyser", type="primary"):

    if not text.strip():
        st.warning("Merci d'entrer un texte avant d'analyser.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(API_URL, json={"text": text}, timeout=10)
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Impossible de contacter l'API. "
                    "Vérifie que le conteneur Docker tourne bien sur le port 8000."
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de l'appel à l'API : {e}")

            else:
                label = result["label"]
                score = result["score"]
                probabilities = result["probabilities"]

                if label == "POSITIVE":
                    st.success(f"Sentiment : **{label}** ({score:.2%} de confiance)")
                else:
                    st.error(f"Sentiment : **{label}** ({score:.2%} de confiance)")

                st.write("Détail des probabilités :")
                st.progress(probabilities["POSITIVE"], text=f"Positif : {probabilities['POSITIVE']:.2%}")
                st.progress(probabilities["NEGATIVE"], text=f"Négatif : {probabilities['NEGATIVE']:.2%}")
