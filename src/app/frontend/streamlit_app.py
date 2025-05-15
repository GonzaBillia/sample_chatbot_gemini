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
from pathlib import Path
from typing import Set

import streamlit as st

# ── Import local packages ───────────────────────────────────────────
SRC_PATH = Path(__file__).parents[2]
sys.path.insert(0, str(SRC_PATH))

from app.core.config import get_settings
from app.langchain.chains import get_qa_chain
from app.database.helpers.record_helper import log_qa_async, update_rating_async

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
    """Render slider+button for `record_id`; hide widget after submission."""

    if record_id in st.session_state[_RATED]:
        return  # Already rated

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
            # 1) Persist in Session State
            st.session_state[_RATED][record_id] = rating_val
            if record_id in st.session_state[_PENDING]:
                st.session_state[_PENDING].remove(record_id)

            # 2) Fire‑and‑forget DB update
            threading.Thread(
                target=lambda: update_rating_async(record_id, rating_val),
                daemon=True,
            ).start()

            # 3) Hide widget immediately
            container.empty()
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

    # Entrada del usuario -------------------------------------------
    user_input = st.chat_input("Haz tu pregunta aquí…")
    rendered_now: Set[uuid.UUID] = set()

    if user_input:
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            spinner = st.empty()
            spinner.write("Pensando…")

            try:
                qa_chain.retriever.search_kwargs["k"] = 5
                result = qa_chain.invoke({"query": user_input})
                answer: str = result["result"]
            except Exception:
                answer = "Lo siento, ocurrió un error al procesar tu pregunta."

            spinner.empty()
            st.markdown(answer)

            # Registrar QA ------------------------------------------
            try:
                rec_id = log_qa_async(
                    question=user_input,
                    answer=answer,
                    qa_chain=qa_chain,
                    user_id="anon",
                )
                st.session_state[_IDS].append(rec_id)
                st.session_state[_PENDING].append(rec_id)

                _render_rating_widget(rec_id)
                rendered_now.add(rec_id)
            except Exception as exc:
                st.warning(f"No se pudo registrar la conversación: {exc}")

        # Guardar en historial --------------------------------------
        st.session_state[_HIST].extend([("user", user_input), ("assistant", answer)])

    # Dibujar widgets pendientes (respuestas de ciclos previos) -----
    remaining_ids: Set[uuid.UUID] = set(st.session_state[_PENDING]) - rendered_now
    for pid in remaining_ids:
        _render_rating_widget(pid)


if __name__ == "__main__":
    main()
