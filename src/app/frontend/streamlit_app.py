# src/app/frontend/streamlit_app.py

import os
import sys
from pathlib import Path

# — Añadir src/ al path para que importe app.* —
SRC_PATH = Path(__file__).parents[2]
sys.path.insert(0, str(SRC_PATH))
# ——————————————————————————————

import streamlit as st
from app.core.config import get_settings
from app.langchain.chains import get_qa_chain

def main():
    # 1) Configuración / API Key
    settings = get_settings()
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    # 2) Layout
    st.set_page_config(page_title="ChatBot Anto", layout="centered")
    st.title("💬 Soy Anto – Asistente FSA")
    st.subheader("Comienza con: ¿Que puedo preguntarte?")

    if st.button("🔄 Nuevo chat"):
        st.session_state.history = []

    # 3) Inicializar chain y estado
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = get_qa_chain()
    if "history" not in st.session_state:
        st.session_state.history = []  # lista de tuplas (rol, mensaje)

    qa_chain = st.session_state.qa_chain

    # 4) Mostrar todo el historial (solo interacciones anteriores)
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    # 5) Leer nueva entrada de usuario
    user_input = st.chat_input("Haz tu pregunta aquí...")

    if user_input:
        # 5.1) Mostrar de inmediato la burbuja del usuario
        st.chat_message("user").write(user_input)

        # 5.2) Mostrar “Procesando…” y llamar al LLM
        with st.chat_message("assistant"):
            # Creamos un placeholder que luego vaciaremos
            placeholder = st.empty()
            placeholder.write("Pensando…")

            qa_chain.retriever.search_kwargs["k"] = 5
            try:
                output = qa_chain.invoke({"query": user_input})
                answer = output["result"]
            except Exception:
                answer = "Lo siento, ocurrió un error al procesar tu pregunta."

            # 5.3) Vaciamos el placeholder para quitar "Procesando…"
            placeholder.empty()
            # 5.4) Escribimos la respuesta en su lugar
            st.write(answer)

        # 5.3) Guardar en historial para futuros reruns
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("assistant", answer))

if __name__ == "__main__":
    main()
