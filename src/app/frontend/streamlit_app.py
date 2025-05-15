# -*- coding: utf-8 -*-
"""
Streamlit frontend for **Anto – Asistente FSA**

• Invoca el qa‑chain de LangChain.
• Registra cada interacción (hilo asíncrono) en `qa.qa_records`.
• Muestra un widget de calificación 0‑10 que se oculta **sin usar** `st.experimental_rerun`.
"""

from __future__ import annotations

import os
import sys
import uuid
import threading
import json
from pathlib import Path
from typing import Set

import streamlit as st

# ── Import local packages ───────────────────────────────────────────
SRC_PATH = Path(__file__).parents[2]
sys.path.insert(0, str(SRC_PATH))

from app.core.config import get_settings
from app.langchain.chains import get_qa_chain
from app.database.helpers.record_helper import log_qa_async, update_rating_async
from app.database.helpers.classification_helper import classification_chain
from app.config.complexity_map import CONFIG_MAP

# ── Session‑state keys ──────────────────────────────────────────────
_HIST = "hist"       # List[tuple[str, str]]
_IDS = "qa_ids"     # List[UUID]
_PENDING = "pending" # List[UUID] awaiting rating
_RATED = "rated"     # Dict[UUID, int]


def _init_state() -> None:
    """Ensure all keys exist in Session State."""
    st.session_state.setdefault(_HIST, [])
    st.session_state.setdefault(_IDS, [])
    st.session_state.setdefault(_PENDING, [])
    st.session_state.setdefault(_RATED, {})


# ── Rating widget ──────────────────────────────────────────────────

def _render_rating_widget(record_id: uuid.UUID) -> None:
    """Render slider+button para `record_id`; tras calificar, desaparece y muestra agradecimiento."""

    # 2) Solo pintamos mientras siga pendiente
    if record_id not in st.session_state[_PENDING]:
        return

    container = st.container()
    slider_key = f"slider_{record_id.hex}"
    button_key = f"btn_{record_id.hex}"

    with container:
        rating_val: int = st.slider(
            "Califica de 0 (pésima) a 10 (excelente)",
            min_value=0,
            max_value=10,
            step=1,
            value=5,
            key=slider_key,
        )
        if st.button("Enviar calificación", key=button_key):
            # 3) Guardar en sesión
            st.session_state[_RATED][record_id] = rating_val
            st.session_state[_PENDING].remove(record_id)

            # 4) Actualizar en segundo plano
            threading.Thread(
                target=lambda: update_rating_async(record_id, rating_val),
                daemon=True,
            ).start()

            st.success("¡Gracias por tu feedback!")


# ── Main app ───────────────────────────────────────────────────────

def main() -> None:
    settings = get_settings()
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    _init_state()

    st.set_page_config(page_title="ChatBot Anto", layout="centered")
    st.title("💬 Soy Anto – Asistente FSA")

    if st.button("🔄 Nuevo chat"):
        for key in (_HIST, _IDS, _PENDING, _RATED):
            st.session_state[key] = [] if key != _RATED else {}
        # El botón de Streamlit ya provoca un rerun automático.

    # QA chain -------------------------------------------------------
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = get_qa_chain()
    qa_chain = st.session_state.qa_chain

    # Mostrar historial ---------------------------------------------
    for role, msg in st.session_state[_HIST]:
        st.chat_message(role).write(msg)

    if st.session_state[_PENDING]:
        last_rec_id = st.session_state[_PENDING][-1]
        _render_rating_widget(last_rec_id)

    # Entrada del usuario -------------------------------------------
    user_input = st.chat_input("Haz tu pregunta aquí…")
    rendered_now: Set[uuid.UUID] = set()

    if user_input:
        st.chat_message("user").write(user_input)

        # 3) Invocar la chain y mostrar respuesta
        with st.chat_message("assistant"):
            spinner = st.empty()
            spinner.write("Pensando…")
            
            # 1) Clasificar complejidad
            output = classification_chain.invoke({"question": user_input})
            complexity = output["text"]

            if complexity not in CONFIG_MAP:
                complexity = "simple"

            params = CONFIG_MAP.get(complexity, CONFIG_MAP["simple"])

            # 2) Reconstruir la QA chain con los parámetros adecuados
            qa_chain = get_qa_chain(
                top_k=params["top_k"],
                temperature=params["temperature"],
            )

            try:
                result = qa_chain.invoke({"query": user_input})
                answer = result["result"]
            except Exception:
                answer = "Lo siento, ocurrió un error al procesar tu pregunta."
            spinner.empty()
            st.markdown(answer)

            # 4) Log y rating
            try:
                rec_id = log_qa_async(
                    user_input,
                    answer,
                    qa_chain,
                    "anon",
                )
                st.session_state[_IDS].append(rec_id)
                st.session_state[_PENDING].append(rec_id)
            
            except Exception as exc:
                st.warning(f"No se pudo registrar la conversación: {exc}")


        # 5) Actualizar historial
        st.session_state[_HIST].extend([
            ("user", user_input),
            ("assistant", answer)
        ])

if __name__ == "__main__":
    main()