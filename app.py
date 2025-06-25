import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import sys

# Descargar recursos si no existen
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Mostrar ruta del script
try:
    ruta_script = os.path.abspath(__file__)
except NameError:
    ruta_script = os.path.abspath(sys.argv[0])

st.subheader("📍 Ruta actual del script ejecutado:")
st.code(ruta_script)

# Función para preprocesar texto
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

# Cargar modelos y vectorizador
vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')
svm_model = joblib.load('./model/svm_model.pkl')
nb_model = joblib.load('./model/naive_bayes_model.pkl')
lr_model = joblib.load('./model/logistic_regression_model.pkl')

# Título de la app
st.title("🧠 Sentiment Analyzer Pro - Versión Modificada 🎯")
st.markdown("Esta aplicación analiza el **sentimiento** de un texto usando modelos de aprendizaje automático. Proyecto modificado para el Diplomado IA por *Deivhy Torres Vargas*.")

# Entrada de texto
input_text = st.text_area("✏️ Ingresa un tweet o comentario para analizar:")

# Selección de modelos
st.write("📌 Selecciona uno o más modelos para el análisis:")
use_nb = st.checkbox('Naive Bayes')
use_svm = st.checkbox('SVM')
use_lr = st.checkbox('Logistic Regression')

# Botón de análisis
if st.button("🔍 Analizar Sentimiento"):
    if not input_text:
        st.warning("⚠️ Por favor, ingresa un texto.")
    elif not (use_nb or use_svm or use_lr):
        st.warning("⚠️ Selecciona al menos un modelo.")
    else:
        input_text_processed = preprocess_text(input_text)
        input_text_vect = vectorizer.transform([input_text_processed])

        st.markdown(f"**Texto ingresado:** `{input_text}`")
        st.divider()

        # Naive Bayes
        if use_nb:
            pred = nb_model.predict(input_text_vect)[0]
            prob = nb_model.predict_proba(input_text_vect)[0]
            st.info(f"📘 Naive Bayes: {'👍 Positivo' if pred == 1 else '👎 Negativo'} (Confianza: {prob[pred]:.2f})")

        # SVM
        if use_svm:
            pred = svm_model.predict(input_text_vect)[0]
            st.info(f"📗 SVM: {'👍 Positivo' if pred == 1 else '👎 Negativo'}")

        # Regresión Logística
        if use_lr:
            pred = lr_model.predict(input_text_vect)[0]
            prob = lr_model.predict_proba(input_text_vect)[0]
            st.info(f"📙 Logistic Regression: {'👍 Positivo' if pred == 1 else '👎 Negativo'} (Confianza: {prob[pred]:.2f})")

# Botón para reiniciar
if st.button("🧹 Limpiar todo"):
    st.experimental_rerun()
